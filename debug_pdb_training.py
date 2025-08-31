import copy
import logging
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utils
import wandb
from etnn.pdbbind.pdbbind import PDBBindCC

# torch.set_float32_matmul_precision("high")  # Use high precision for matmul
os.environ["WANDB__SERVICE_WAIT"] = "600"

SPLIT_OOD = True


logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, model, opt, sched, force_restart):
    best_model = copy.deepcopy(model)
    device = next(model.parameters()).device
    if not force_restart and os.path.isfile(checkpoint_path):
        print("Loading model from checkpoint!")
        checkpoint = torch.load(checkpoint_path)
        model.to("cpu")
        best_model.to("cpu")
        model.load_state_dict(checkpoint["model"])
        best_model.load_state_dict(checkpoint["best_model"])
        best_pcc = checkpoint["best_pcc"]
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        model.to(device)
        best_model.to(device)
        
        # BUG FIX ATTEMPT: Ensure optimizer state is on the correct device
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        return checkpoint["epoch"], checkpoint["run_id"], best_model, best_pcc
    else:
        return 0, None, best_model, 0



def make_protein_ood_split(dataset, val_frac=0.2, seed=42):
    N = len(dataset)
    prot_ids = []
    for i in range(N):
        g = dataset[i]
        pid = None
        if hasattr(g, "id") and g.id is not None:
            # expected format: TARGETCHEMBLID_MOLECULECHEMBLID
            pid = str(g.id).split("_", 1)[0]  # protein side
        # (optionally: fallbacks if you later add attributes like g.uniprot)
        prot_ids.append(pid)

    unique_proteins = sorted({p for p in prot_ids if p is not None})
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_proteins)
    n_val = max(1, int(len(unique_proteins) * val_frac))
    val_proteins = set(unique_proteins[:n_val])
    is_val = np.array([(p in val_proteins) if p is not None else False for p in prot_ids])
    val_idx = np.nonzero(is_val)[0]
    trn_idx = np.nonzero(~is_val)[0]

    # sanity check: disjoint proteins
    trn_prots = {prot_ids[i] for i in trn_idx if prot_ids[i] is not None}
    val_prots = {prot_ids[i] for i in val_idx if prot_ids[i] is not None}
    assert trn_prots.isdisjoint(val_prots), "Protein overlap between train and val!"

    train_subset = dataset.index_select(trn_idx)
    valid_subset = dataset.index_select(val_idx)

    print(f"[protein-OOD] train_samples={len(train_subset)}, val_samples={len(valid_subset)}, \n"
          f"train_unique_proteins={len(trn_prots)}, val_unique_proteins={len(val_proteins)}")
    
    return train_subset, valid_subset

def current_lr(opt):
    return opt.param_groups[0]["lr"]

def evaluate(cfg: DictConfig, model, test_dataloader, device, mad, mean):
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ==== Evaluation ====
    logging.info(f"Evaluating model...\nTest samples:  {len(test_dataloader.dataset)}")
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
    #logging.info("Denormalizing predictions for evaluation")
    mean_cpu = mean.detach().cpu()
    mad_cpu = mad.detach().cpu()
    denorm_preds = preds * mad_cpu + mean_cpu
    #denorm_preds = preds

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


@hydra.main(config_path="conf/conf_pdb", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # ==== Initial setup =====
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ==== Get dataset and loader ======
    logger.info("Loading dataset...")
    train_dataset = PDBBindCC(
        index=cfg.dataset.index,
        root=f"data/pdbbind/{cfg.dataset_name}",
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        supercell=cfg.dataset.supercell,
        connect_cross=cfg.dataset.connect_cross,
        r_cut=cfg.dataset.r_cut,
        force_reload=cfg.dataset.force_reload if 'force_reload' in cfg else False,
    )
    
    #if SPLIT_OOD:
    #    train_subset, valid_subset = make_protein_ood_split(train_dataset)
    #else:
    #    # deterministic 80/20 split
    #    rng = np.random.default_rng(cfg.seed)
    #    all_idx = np.arange(len(train_dataset), dtype=np.int64)
    #    rng.shuffle(all_idx)
    #    n_val = int(0.2 * len(all_idx))
    #    val_idx = all_idx[:n_val]
    #    train_idx = all_idx[n_val:]
    #    train_subset = train_dataset.index_select(train_idx)
    #    valid_subset = train_dataset.index_select(val_idx)



    logger.info("Dataset loaded!")

    # ==== Get model =====
    logger.info("Loading Model...")
    model = utils.get_model(cfg, train_dataset)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    #logger.info(model)
    logger.info("Model loaded!")

    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    #valid_dataloader = DataLoader(
    #    valid_subset,
    #    num_workers=cfg.training.num_workers,
    #    pin_memory=cfg.training.pin_memory,
    #    batch_size=cfg.training.batch_size,
    #    shuffle=False,
    #)


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
    
    logger.info(f"Mean: {mean}, Mad: {mad}")

    # ==== Get optimization objects =====
    if cfg.training.crit == 'L1Loss':
        crit = torch.nn.L1Loss(reduction="mean")
    else:
        crit = torch.nn.MSELoss()
    opt_kwargs = dict(lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    opt = torch.optim.Adam(model.parameters(), **opt_kwargs)

    # Choose scheduler mode
    try:
        scheduler_mode = cfg.training.scheduler_mode
    except Exception as e:
        raise ValueError("Invalid scheduler mode")

    def _noop_sched(optimizer):
        class _NoOp:
            def step(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, _): pass
            def get_last_lr(self): return [optimizer.param_groups[0]["lr"]]
        return _NoOp()

    print("using constant LR (no scheduler)")
    sched = _noop_sched(opt)

    best_pcc = 0


    # === Configure checkpoint and wandb logging ===
    ckpt_filename = f"{cfg.experiment_name}__{cfg.dataset_name}.pth"
    if cfg.ckpt_prefix is not None:
        ckpt_filename = f"{cfg.ckpt_prefix}_{ckpt_filename}"
    checkpoint_path = f"casf_val/{cfg.ckpt_dir}/{ckpt_filename}"
    logging.info(f"Checkpoint path set as: {checkpoint_path}")

    start_epoch, run_id, best_model, best_pcc = load_checkpoint(
        checkpoint_path, model, opt, sched, cfg.force_restart
    )


    if start_epoch >= cfg.training.epochs:
        logger.info("Training already completed. Exiting.")
        return


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
            try:
                pred = model(batch)
            except Exception as e:
                raise Exception(f"Forward pass failed for id's {batch.id}")
            loss = crit(pred, (batch.y - mean) / mad)
            
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
        
        epoch_iter.set_postfix(train_mae=epoch_mae_train, val_mae=epoch_mae_val, val_mse=epoch_mse_val)
        

            




if __name__ == "__main__":
    main()
