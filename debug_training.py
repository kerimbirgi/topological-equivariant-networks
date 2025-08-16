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

def global_grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total += float(param_norm.item() ** 2)
    return total ** 0.5

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
    logger.debug("Imports successful and program started")
    
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    logger.info(model)

    train_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'train_indices.npy'))
    valid_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'val_indices.npy'))
    test_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'test_indices.npy'))

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
    if cfg.use_single_batch:
        one_batch = next(iter(train_dataloader))
        train_dataloader = [one_batch]
        y1 = one_batch.y.to(device)
        mean = y1.mean()
        mad = (y1 - mean).abs().mean().clamp_min(1e-8)
    else:
        mean, mad = utils.calc_mean_mad(train_dataloader)
        mean, mad = mean.to(device), mad.to(device)

    # ==== Get optimization objects =====
    crit = torch.nn.L1Loss(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,
                       weight_decay=cfg.training.weight_decay)
    best_loss = float("inf")


    # === Configure checkpoint and wandb logging ===
    ckpt_filename = f"{cfg.experiment_name}__{cfg.dataset_name}.pth"
    if cfg.ckpt_prefix is not None:
        ckpt_filename = f"{cfg.ckpt_prefix}_{ckpt_filename}"
    checkpoint_path = f"{cfg.ckpt_dir}/{ckpt_filename}"
    logging.info(f"Checkpoint path set as: {checkpoint_path}")


    start_epoch = 0

    run_id = ckpt_filename.split(".")[0] + "__" + wandb.util.generate_id()
    if cfg.ckpt_prefix is not None:
        run_id = "__".join([cfg.ckpt_prefix, run_id])

    # create wandb config and add number of parameters
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_config["num_params"] = num_params

    wandb.init(
        project="bindingnet_debug",
        name=f"{cfg.experiment_name}_{cfg.dataset_name}",
        entity=os.environ.get("WANDB_ENTITY"),
        config=wandb_config,
        id=run_id,
        resume="allow",
    )


    # === debug loop ===
    global_step = start_epoch * len(train_dataloader)
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

            # grad norms
            preclip = global_grad_norm(model.parameters())
    
            clipped_flag = 0
            postclip = preclip
            if cfg.training.clip_gradients:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.clip_amount
                )
                # torch returns the *pre-clip* norm; keep it for logging
                postclip = global_grad_norm(model.parameters())
                clipped_flag = int(total_norm.item() > cfg.training.clip_amount)

            opt.step()
            current_lr = opt.param_groups[0]["lr"]

            # per-step logging
            wandb.log({
                "step/loss": loss.item(),
                "step/mae": mae.item(),
                "step/grad_norm_preclip": preclip,
                "step/grad_norm_postclip": postclip,
                "step/grad_clipped": clipped_flag,
                "step/lr": current_lr,
            }, step=global_step)
            global_step += 1

            batch_iter.set_postfix(loss=float(loss.item()), lr=current_lr)  
        

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        epoch_iter.set_postfix(train_mae=epoch_mae_train, val_mae=epoch_mae_val)



if __name__ == "__main__":
    main()
