import copy
import logging
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utils_head as utils
import wandb
from etnn.bindingnet.bindingnetcc import BindingNetCC

# torch.set_float32_matmul_precision("high")  # Use high precision for matmul
os.environ["WANDB__SERVICE_WAIT"] = "600"


logger = logging.getLogger(__name__)

def current_lr(opt):
    return opt.param_groups[0]["lr"]

def setup_adamw(cfg, model):
    decay, no_decay = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if any(k in n.lower() for k in ["bias","norm","layernorm","batchnorm","bn"]):
            no_decay.append(p)
        else:
            decay.append(p)

    wd = float(cfg.training.weight_decay)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.training.lr, betas=(0.9, 0.98)
    )
    return opt

def build_scheduler(cfg, scheduler_mode, opt):
    def _noop_sched(optimizer):
        class _NoOp:
            def step(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, _): pass
            def get_last_lr(self): return [optimizer.param_groups[0]["lr"]]
        return _NoOp()

    if scheduler_mode == "none":
        print("using constant LR (no scheduler)")
        sched = _noop_sched(opt)
    elif scheduler_mode == 'cosine_warmup':
        print("using warmup + cosine (LambdaLR)")
        warm = int(getattr(cfg.training, "warmup_epochs", 2))
        total = int(cfg.training.epochs)
        eta_min = float(cfg.training.min_lr)
        base_lr = float(cfg.training.lr)

        def lr_lambda(epoch):
            # epoch is 0-based
            e = epoch + 1
            if warm > 0 and e <= warm:
                return max(1e-6, e / warm)  # linear warmup 0→1 (clamped >0 for safety)
            # cosine from warm+1 → total
            # scale from base_lr to eta_min
            t = max(1, total - max(warm, 0))
            k = e - max(warm, 0)
            cos = 0.5 * (1.0 + np.cos(np.pi * min(k, t) / t))
            # return factor relative to base_lr so Optimizer LR = base_lr * factor
            return (eta_min / base_lr) + (1.0 - (eta_min / base_lr)) * cos

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif scheduler_mode == 'sgdr':
        print("using cosine annealing with warm restarts (SGDR)")
        num_cycles = max(1, int(getattr(cfg.training, "num_lr_cycles", 3)))
        T_0 = max(1, cfg.training.epochs // num_cycles)
        T_mult = int(getattr(cfg.training, "T_mult", 2))
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=T_0, T_mult=T_mult, eta_min=cfg.training.min_lr
        )
    elif scheduler_mode == 'cosine_annealing':
        print("using cosineannealinglr")
        cycles = max(1, int(getattr(cfg.training, "num_lr_cycles", 1)))
        T_max = max(1, cfg.training.epochs // cycles)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max, eta_min=cfg.training.min_lr
        )
    elif scheduler_mode == 'ReduceLROnPlateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            patience=5,     
        )
    else:
        raise ValueError(f"Unknown training.scheduler='{scheduler_mode}'")

    return sched

def evaluate(cfg: DictConfig, model, test_dataloader, device, mad, mean):
    # ==== Evaluation ====
    model.eval()
    preds_cpu: list[torch.Tensor] = []
    targets_cpu: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            preds_cpu.append(pred.detach().cpu())
            targets_cpu.append(batch.y.detach().cpu())

    preds = torch.cat(preds_cpu)   # on CPU
    targets = torch.cat(targets_cpu)  # on CPU

    # Always denormalize predictions for evaluation (since we always normalize during training)
    # if normalization is off mean_cpu=0, mad_cpu=1 
    mean_cpu = mean.detach().cpu()
    mad_cpu = mad.detach().cpu()
    denorm_preds = preds * mad_cpu + mean_cpu

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
    pcc, pcc_pvalue = pearsonr(denorm_preds, targets)
    tau, tau_pvalue = kendalltau(denorm_preds, targets, variant="b")  # (tau, p-value)
    predictions_range = f"[{torch.min(denorm_preds)}, {torch.max(denorm_preds)}]"
    targets_range = f"[{torch.min(targets)}, {torch.max(targets)}]"

    with open(os.path.join(cfg.results_dir, f'{cfg.experiment_name}_{cfg.dataset_name}_evaluation.txt'), 'w') as f:
        f.write(f"Test MAE: {mae.item()}\n")
        f.write(f"Test MSE: {mse.item()}\n")
        f.write(f"Test RMSE: {rmse.item()}\n")
        f.write(f"Test Pearson correlation coefficient (PCC): {pcc}, PCC p-value: {pcc_pvalue}\n")
        f.write(f"Test Kendall-Tau: {tau}, Kendall-Tau p-value: {tau_pvalue}\n")
        f.write(f"Normalization parameters - MAD: {mad_cpu}, Mean: {mean_cpu}\n")
        f.write(f"Predictions range: {predictions_range}\n")
        f.write(f"Targets range: {targets_range}\n")

    logger.info(f"Test MAE: {mae.item()}")
    logger.info(f"Test MSE: {mse.item()}")
    logger.info(f"Test RMSE: {rmse.item()}")
    logger.info(f"Test Pearson correlation coefficient (PCC): {pcc}, PCC p-value: {pcc_pvalue}")
    logger.info(f"Test Kendall-Tau: {tau}, Kendall-Tau p-value: {tau_pvalue}")
    logger.info(f"Predictions range: {predictions_range}")
    logger.info(f"Targets range: {targets_range}")
    logger.info(f"Normalization parameters - MAD: {mad_cpu}, Mean: {mean_cpu}")


@hydra.main(config_path="conf/conf_bindingnet", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.debug("Imports successful and program started")
    
    # ==== Initial setup =====
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ==== Get dataset and loader ======
    logger.info("Loading dataset...")
    dataset = BindingNetCC(
        index=cfg.dataset.index,
        root=f"data/bindingnetcc/{cfg.dataset_name}",
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        supercell=cfg.dataset.supercell,
        connect_cross=cfg.dataset.connect_cross,
        r_cut=cfg.dataset.r_cut,
        force_reload=cfg.dataset.force_reload if 'force_reload' in cfg else False,
    )
    logger.info("Dataset loaded!")

    # ==== Get model =====
    logger.info("Loading Model...")
    model = utils.get_model(cfg, dataset)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    #logger.info(model)
    logger.info("Model loaded!")


    # Get train/test splits
    indices_files = os.listdir(cfg.dataset.splits_dir)
    cleanup = (
        'train_etnn_filtered.npy' not in indices_files or 
        'val_etnn_filtered.npy' not in indices_files or 
        'test_etnn_filtered.npy' not in indices_files
    )

    if cleanup or cfg.dataset.filter_indices:
        train_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'train_indices.npy'))
        val_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'val_indices.npy'))
        test_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'test_indices.npy'))
        # Clean up dataset to only include valid tuples
        df = pd.read_csv(cfg.dataset.index)
        kept_mask = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering indices"):
            tuple_id = row['Target ChEMBLID'] + '_' + row['Molecule ChEMBLID']
            merged_data_path = os.path.join(dataset.root, 'preprocessed/merged', f'{tuple_id}.pt')
            kept_mask.append(os.path.exists(merged_data_path))
        kept_mask = np.array(kept_mask, dtype=bool)

        # Map original to compacted indices
        print("Remapping indices...")
        compacted = np.cumsum(kept_mask) - 1  # valid where kept_mask is True
        def remap(orig_idx: np.ndarray) -> np.ndarray:
            idx = np.asarray(orig_idx, dtype=np.int64).reshape(-1)
            valid = kept_mask[idx]
            return compacted[idx[valid]].astype(np.int64)
        train_sel = remap(train_indices)
        valid_sel = remap(val_indices)
        test_sel  = remap(test_indices)
        print("Remapping done!")

        np.save(os.path.join(cfg.dataset.splits_dir, 'train_etnn_filtered.npy'), train_sel)
        np.save(os.path.join(cfg.dataset.splits_dir, 'val_etnn_filtered.npy'), valid_sel)
        np.save(os.path.join(cfg.dataset.splits_dir, 'test_etnn_filtered.npy'), test_sel)
        print("saved new indices to:")
        print(cfg.dataset.splits_dir)
        train_indices = train_sel
        val_indices = valid_sel
        test_indices = test_sel
    else:
        train_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'train_etnn_filtered.npy'))
        val_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'val_etnn_filtered.npy'))
        test_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'test_etnn_filtered.npy'))

    train_subset = dataset.index_select(train_indices)
    valid_subset = dataset.index_select(val_indices)
    test_subset = dataset.index_select(test_indices)

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
    if cfg.training.normalize_targets:
        logger.info("Normalization On!")
        mean, mad = utils.calc_mean_mad(train_dataloader)
        mean, mad = mean.to(device), mad.to(device)
    else:
        logger.info("Normalization Off!")
        # normalization/denormalization return identity this way
        mean = torch.tensor([0.0], device=device)
        mad  = torch.tensor([1.0], device=device)

    # ==== Get optimization objects =====
    if cfg.training.crit == 'L1Loss':
        crit = torch.nn.L1Loss(reduction="mean")
    else:
        crit = torch.nn.MSELoss()
    opt_kwargs = dict(lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    if cfg.training.optimizer == 'Adam':
        opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
    elif cfg.training.optimizer == 'AdamW':
        opt = setup_adamw(cfg, model)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")


    # Choose scheduler mode
    try:
        scheduler_mode = cfg.training.scheduler_mode
    except Exception as e:
        raise ValueError("Invalid scheduler mode")

    sched = build_scheduler(cfg, scheduler_mode, opt)

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

    if cfg.training.continue_from_weights:
        logging.info(f"Starting with pretrained weights from {cfg.training.start_checkpoint}")
        start_ckpt_path = f"{cfg.ckpt_dir}/{cfg.training.start_checkpoint}.pth"
        start_checkpoint = torch.load(start_ckpt_path, weights_only=False)
        model.load_state_dict(start_checkpoint["best_model"])

        opt = setup_adamw(cfg, model)
        sched = build_scheduler(cfg, scheduler_mode, opt)
        start_epoch = 0
        best_loss = float("inf")
        best_model = copy.deepcopy(model)
        run_id = None  # start a fresh W&B run


    # ==== If eval only, evaluate and exit ====
    if cfg.eval_only:
        logger.info("Running evaluation only with best model")
        evaluate(cfg, best_model, test_dataloader, device, mad, mean)
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
    early_stop_counter = 0
    epoch_iter = tqdm(range(start_epoch, cfg.training.epochs), desc="Epochs", position=0)
    for epoch in epoch_iter:
        epoch_start_time, epoch_mae_train, epoch_mse_train, epoch_loss_train, epoch_mae_val, epoch_mse_val = time.time(), 0, 0, 0, 0, 0

        model.train()
        batch_iter = tqdm(train_dataloader, desc=f"Train {epoch+1}/{num_epochs}", position=1, leave=False)
        for batch in batch_iter:
            opt.zero_grad(set_to_none=True)
            batch = batch.to(device)

            pred = model(batch)
            loss = crit(pred, (batch.y - mean) / mad)
            #mae = crit(pred * mad + mean, batch.y)
            denorm_pred = pred * mad + mean
            mae = torch.nn.functional.l1_loss(denorm_pred, batch.y)
            mse = torch.nn.functional.mse_loss(denorm_pred, batch.y)
            loss.backward()

            if cfg.training.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.clip_amount
                )

            opt.step()

            epoch_loss_train += loss.item()
            epoch_mae_train += mae.item()
            epoch_mse_train += mse.item()


            batch_iter.set_postfix(loss=float(loss.item()), lr=current_lr(opt))  
        

        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(valid_dataloader):
                batch = batch.to(device)
                pred = model(batch)
                
                # Always denormalize for proper validation metrics (since we always normalize during training)
                denorm_pred = pred * mad + mean
                val_mae = torch.nn.functional.l1_loss(denorm_pred, batch.y)
                val_mse = torch.nn.functional.mse_loss(denorm_pred, batch.y)
                #mae = crit(denorm_pred, batch.y)
                epoch_mae_val += val_mae.item()
                epoch_mse_val += val_mse.item()

        epoch_mae_train /= len(train_dataloader)
        epoch_loss_train /= len(train_dataloader)
        epoch_mse_train /= len(train_dataloader)
        epoch_mae_val /= len(valid_dataloader)
        epoch_mse_val /= len(valid_dataloader)

        if cfg.training.scheduler_mode == 'ReduceLROnPlateau':
            sched.step(epoch_mae_val)
        else:
            sched.step()

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
                "Train MSE": epoch_mse_train,
                "Validation MAE": epoch_mae_val,
                "Validation MSE": epoch_mse_val,
                "Epoch Duration": epoch_duration,
                "Learning Rate": current_lr(opt),
            },
            step=epoch,
        )
        epoch_iter.set_postfix(train_mae=epoch_mae_train, val_mae=epoch_mae_val)

        if cfg.training.early_stop:
            if epoch_mae_val > best_loss:
                early_stop_counter +=1
            else:
                early_stop_counter = 0
            
            if early_stop_counter > cfg.training.early_stop_patience:
                break # Finish training early if patience has been surpassed

    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    utils.save_checkpoint(
        path=checkpoint_path,
        model=model,
        best_model=best_model,
        best_loss=best_loss,
        opt=opt,
        sched=sched,
        epoch=cfg.training.epochs - 1,
        run_id=run_id,
    )

    # ==== Final test evaluation after training completion ====
    logger.info("Training completed. Running final evaluation on test set...")

    logger.info("Running evaluation with current model")
    evaluate(cfg, model, test_dataloader, device, mad, mean)

    logger.info("Running evaluation with best model")
    evaluate(cfg, best_model, test_dataloader, device, mad, mean)
    
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
