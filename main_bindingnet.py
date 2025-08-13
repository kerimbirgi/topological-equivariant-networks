import copy
import logging
import os
import time
import numpy as np
import pandas as pd

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utils
import wandb
from etnn.bindingnet.bindingnetcc import BindingNetCC

# torch.set_float32_matmul_precision("high")  # Use high precision for matmul
os.environ["WANDB__SERVICE_WAIT"] = "600"


logger = logging.getLogger(__name__)

def evaluate(cfg: DictConfig, model, test_dataloader, device, mad, mean):
    # ==== Evaluation ====
    model.eval()
    preds_cpu: list[torch.Tensor] = []
    targets_cpu: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in test_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            preds_cpu.append(pred.detach().cpu())
            targets_cpu.append(batch.y.detach().cpu())

    preds = torch.cat(preds_cpu)   # on CPU
    targets = torch.cat(targets_cpu)  # on CPU

    # Denormalize on CPU only if normalization was used
    if cfg.training.normalize_targets:
        logging.info("Using normalized targets")
        mean_cpu = mean.detach().cpu()
        mad_cpu = mad.detach().cpu()
        denorm_preds = preds * mad_cpu + mean_cpu
    else:
        logging.info("Using unnormalized targets")
        denorm_preds = preds

    # Save predictions/targets to CSV as plain numeric arrays
    df = pd.DataFrame({
        'predictions': denorm_preds.numpy().ravel(),
        'targets': targets.numpy().ravel(),
    })
    df.to_csv(os.path.join(cfg.results_dir, f'{cfg.experiment_name}_{cfg.dataset_name}_predictions.csv'), index=False)

    # Compute metrics on CPU
    mae = torch.nn.functional.l1_loss(denorm_preds, targets, reduction='mean')
    mse = torch.nn.functional.mse_loss(denorm_preds, targets, reduction='mean')
    rmse = torch.sqrt(mse)
    predictions_range = f"[{torch.max(denorm_preds)}, {torch.min(denorm_preds)}]"
    targets_range = f"[{torch.max(targets)}, {torch.min(targets)}]"
    with open(os.path.join(cfg.results_dir, f'{cfg.experiment_name}_{cfg.dataset_name}_evaluation.txt'), 'w') as f:
        f.write(f"Test MAE: {mae.item()}\n")
        f.write(f"Test MSE: {mse.item()}\n")
        f.write(f"Test RMSE: {rmse.item()}\n")
        f.write(f"Predictions range: {predictions_range}\n")
        f.write(f"Targets range: {targets_range}\n")

    logger.info(f"Test MAE: {mae.item()}")
    logger.info(f"Test MSE: {mse.item()}")
    logger.info(f"Test RMSE: {rmse.item()}")
    logger.info(f"Predictions range: {predictions_range}")
    logger.info(f"Targets range: {targets_range}")


@hydra.main(config_path="conf/conf_bindingnet", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # ==== Initial setup =====
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ==== Get dataset and loader ======
    dataset = BindingNetCC(
        index=cfg.dataset.index,
        root=f"data/bindingnetcc/{cfg.dataset_name}",
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        supercell=cfg.dataset.supercell,
        connect_cross=cfg.dataset.connect_cross,
        r_cut=cfg.dataset.r_cut,
        force_reload=cfg.force_reload if 'force_reload' in cfg else False,
    )

    # ==== Get model =====
    model = utils.get_model(cfg, dataset)
    model = model.to(device)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    logger.info(model)

    # Get train/test splits using the original egnn splits for reference
    if cfg.dataset.use_postprocessed:
        train_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'train_sel_postprocessed.npy'))
        valid_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'val_sel_postprocessed.npy'))
        test_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'test_sel_postprocessed.npy'))
    else:
        train_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'train_indices.npy'))
        valid_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'val_indices.npy'))
        test_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'test_indices.npy'))

    if cfg.dataset.cleanup_postprocess:
        # Clean up dataset to only include valid tuples
        df = pd.read_csv(dataset.index)
        kept_mask = []
        for _, row in df.iterrows():
            tuple_id = row['Target ChEMBLID'] + '_' + row['Molecule ChEMBLID']
            merged_data_path = os.path.join(dataset.root, 'preprocessed/merged', f'{tuple_id}.pt')
            kept_mask.append(os.path.exists(merged_data_path))
        kept_mask = np.array(kept_mask, dtype=bool)

        # Map original to compacted indices
        compacted = np.cumsum(kept_mask) - 1  # valid where kept_mask is True
        def remap(orig_idx: np.ndarray) -> np.ndarray:
            idx = np.asarray(orig_idx, dtype=np.int64).reshape(-1)
            valid = kept_mask[idx]
            return compacted[idx[valid]].astype(np.int64)
        train_sel = remap(train_indices)
        valid_sel = remap(valid_indices)
        test_sel  = remap(test_indices)

        np.save(os.path.join(cfg.dataset.splits_dir, 'train_sel_postprocessed.npy'), train_sel)
        np.save(os.path.join(cfg.dataset.splits_dir, 'val_sel_postprocessed.npy'), valid_sel)
        np.save(os.path.join(cfg.dataset.splits_dir, 'test_sel_postprocessed.npy'), test_sel)
        print("saved new indices to:")
        print(cfg.dataset.splits_dir)
    else:
        # keep things as they are since cleanup already done or not necessary
        train_sel = train_indices
        valid_sel = valid_indices
        test_sel = test_indices

    train_subset = dataset.index_select(train_sel)
    valid_subset = dataset.index_select(valid_sel)
    test_subset = dataset.index_select(test_sel)

    print(f"Length of train dataset: {len(train_subset)}")
    print(f"Length of validation dataset: {len(valid_subset)}")
    print(f"Length of test dataset: {len(test_subset)}")

    train_dataloader = DataLoader(
        train_subset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_subset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_subset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # Precompute average deviation of target in training dataloader
    mean, mad = utils.calc_mean_mad(train_dataloader)
    mean, mad = mean.to(device), mad.to(device)

    # ==== Get optimization objects =====
    crit = torch.nn.L1Loss(reduction="mean")
    opt_kwargs = dict(lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
    if cfg.training.lambdalr:
        print("using lambdalr")
        warm_epochs = 5                       # 5 % of 100 epochs
        total_epochs = cfg.training.epochs

        def lr_lambda(epoch):
            if epoch < warm_epochs:
                return (epoch + 1) / warm_epochs            # linear warm-up 0→1
            # cosine part: epoch ∈ [warm, total)
            cosine_e = epoch - warm_epochs
            cosine_T = total_epochs - warm_epochs
            return 0.5 * (1 + np.cos(np.pi * cosine_e / cosine_T))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        print("using cosineannealinglr")
        T_max = cfg.training.epochs // cfg.training.num_lr_cycles
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max, eta_min=cfg.training.min_lr
        )
    best_loss = float("inf")


    # === Configure checkpoint and wandb logging ===
    ckpt_filename = f"{cfg.experiment_name}__{cfg.dataset_name}.pth"
    if cfg.ckpt_prefix is not None:
        ckpt_filename = f"{cfg.ckpt_prefix}_{ckpt_filename}"
    checkpoint_path = f"{cfg.ckpt_dir}/{ckpt_filename}"
    logging.info(f"Checkpoint path set as: {checkpoint_path}")

    start_epoch, run_id, best_model, best_loss = utils.load_checkpoint(
        checkpoint_path, model, opt, sched, cfg.force_restart
    )

    # ==== If eval only, evaluate and exit ====
    if cfg.eval_only:
        logger.info("Running evaluation only")
        evaluate(cfg, model, test_dataloader, device, mad, mean)
        return
    # ==== otherwise continue training ====

    if start_epoch >= cfg.training.epochs:
        logger.info("Training already completed. Exiting.")
        return

    # init wandb logger
    if run_id is None:
        run_id = ckpt_filename.split(".")[0] + "__" + wandb.util.generate_id()
        if cfg.ckpt_prefix is not None:
            run_id = "__".join([cfg.ckpt_prefix, run_id])

    # create wandb config and add number of parameters
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_config["num_params"] = num_params

    wandb.init(
        project="bindingnet_regression",
        name=f"{cfg.experiment_name}_{cfg.dataset_name}",
        entity=os.environ.get("WANDB_ENTITY"),
        config=wandb_config,
        id=run_id,
        resume="allow",
    )

    # === Training loop ===
    num_epochs=cfg.training.epochs
    epoch_iter = tqdm(range(start_epoch, cfg.training.epochs), desc="Epochs", position=0)
    for epoch in epoch_iter:
        epoch_start_time, epoch_mae_train, epoch_loss_train, epoch_mae_val = time.time(), 0, 0, 0

        model.train()
        batch_iter = tqdm(train_dataloader, desc=f"Train {epoch+1}/{num_epochs}", position=1, leave=False)
        for batch in batch_iter:
            opt.zero_grad()
            batch = batch.to(device)

            pred = model(batch)
            loss = crit(pred, (batch.y - mean) / mad)
            mae = crit(pred * mad + mean, batch.y)
            loss.backward()

            if cfg.training.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.clip_amount
                )

            opt.step()

            epoch_loss_train += loss.item()
            epoch_mae_train += mae.item()
            batch_iter.set_postfix(loss=float(loss.item()), lr=sched.get_last_lr()[0])  
        

        sched.step()
        model.eval()
        for _, batch in enumerate(valid_dataloader):
            batch = batch.to(device)
            pred = model(batch)
            
            if cfg.training.normalize_targets:
                mae = crit(pred * mad + mean, batch.y)
            else:
                mae = crit(pred, batch.y)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(train_dataloader)
        epoch_mae_val /= len(valid_dataloader)
        epoch_loss_train /= len(train_dataloader)

        if epoch_mae_val < best_loss:
            best_loss = epoch_mae_val
            best_model = copy.deepcopy(model)

        # Save checkpoint
        if epoch % cfg.training.save_interval == 0:
            logger.info(f"Saving checkpoint at epoch {epoch + 1}")
            utils.save_checkpoint(
                path=checkpoint_path,
                model=model,
                best_model=best_model,
                best_loss=best_loss,
                opt=opt,
                sched=sched,
                epoch=epoch,
                run_id=run_id,
            )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        wandb.log(
            {
                "Train Loss": epoch_loss_train,
                "Train MAE": epoch_mae_train,
                "Validation MAE": epoch_mae_val,
                "Epoch Duration": epoch_duration,
                "Learning Rate": sched.get_last_lr()[0],
            },
            step=epoch,
        )
        epoch_iter.set_postfix(train_mae=epoch_mae_train, val_mae=epoch_mae_val)



if __name__ == "__main__":
    main()
