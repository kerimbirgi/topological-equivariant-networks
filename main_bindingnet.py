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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    logger.info(model)

    # Get train/test splits using the original egnn splits for reference
    train_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'train_indices.npy'))
    valid_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'val_indices.npy'))
    test_indices = np.load(os.path.join(cfg.dataset.splits_dir, 'test_indices.npy'))

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

    train_dataloader = DataLoader(
        dataset.index_select(train_sel),
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        dataset.index_select(valid_sel),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset.index_select(test_sel),
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

    start_epoch, run_id, best_model, best_loss = utils.load_checkpoint(
        checkpoint_path, model, opt, sched, cfg.force_restart
    )

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
    for epoch in tqdm(range(start_epoch, cfg.training.epochs)):
        epoch_start_time, epoch_mae_train, epoch_loss_train, epoch_mae_val = time.time(), 0, 0, 0

        model.train()
        for _, batch in enumerate(train_dataloader):
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
        

        sched.step()
        model.eval()
        for _, batch in enumerate(valid_dataloader):
            batch = batch.to(device)
            pred = model(batch)
            mae = crit(pred * mad + mean, batch.y)

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



if __name__ == "__main__":
    main()
