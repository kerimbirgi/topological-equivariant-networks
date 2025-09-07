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

import utils_head as utils
import wandb
from etnn.pdbbind.pdbbind import PDBBindCC

# torch.set_float32_matmul_precision("high")  # Use high precision for matmul
os.environ["WANDB__SERVICE_WAIT"] = "600"

SPLIT_OOD = True


logger = logging.getLogger(__name__)

def load_best_checkpoint(checkpoint_path, model):
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path)
    model.to("cpu")
    model.load_state_dict(checkpoint["best_model"])
    model.to(device)
    return model


@torch.no_grad()
def eval_with_maps(model, batch):
    # 1) predictions + attention maps (no grad)
    pred, maps = model.predict_with_maps(batch)   # pred: [B], maps per rank: {'weights','batch'}
    return pred.squeeze(-1) if pred.ndim == 2 else pred, maps



def evaluate(cfg: DictConfig, model, test_dataloader, device, results_dir = "final_results"):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, f'{cfg.experiment_name}_{cfg.dataset_name}'), exist_ok=True)
    output_dir = os.path.join(results_dir, f'{cfg.experiment_name}_{cfg.dataset_name}')

    # ==== Evaluation ====
    logging.info(f"Evaluating model...\nTest samples:  {len(test_dataloader.dataset)}")
    model.eval()
    preds_cpu: list[torch.Tensor] = []
    targets_cpu: list[torch.Tensor] = []
    tuple_ids_cpu: list[str] = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            tuple_ids_cpu.append(batch.id.detach().cpu())
            preds_cpu.append(pred.detach().cpu())
            targets_cpu.append(batch.y)

    preds = torch.cat(preds_cpu)   # on CPU
    targets = torch.cat(targets_cpu)  # on CPU

    # Save predictions/targets to CSV as plain numeric arrays
    df = pd.DataFrame({
        'tuple_ids': tuple_ids_cpu,
        'predictions': preds.numpy().ravel(),
        'targets': targets.numpy().ravel(),
    })
    df.to_csv(os.path.join(output_dir, f'{cfg.experiment_name}_{cfg.dataset_name}_predictions.csv'), index=False)

    # Compute metrics on CPU
    mae = torch.nn.functional.l1_loss(preds, targets, reduction='mean')
    mse = torch.nn.functional.mse_loss(preds, targets, reduction='mean')
    rmse = torch.sqrt(mse)
    pcc, pcc_pvalue = pearsonr(preds, targets)
    tau, tau_pvalue = kendalltau(preds, targets, variant="b")  # (tau, p-value)

    with open(os.path.join(output_dir, f'{cfg.experiment_name}_{cfg.dataset_name}_evaluation.txt'), 'w') as f:
        f.write(f"Test MAE: {mae.item()}\n")
        f.write(f"Test MSE: {mse.item()}\n")
        f.write(f"Test RMSE: {rmse.item()}\n")
        f.write(f"Test Pearson correlation coefficient (PCC): {pcc}, PCC p-value: {pcc_pvalue}\n")
        f.write(f"Test Kendall-Tau: {tau}, Kendall-Tau p-value: {tau_pvalue}\n")

    logger.info(f"Test MAE: {mae.item()}")
    logger.info(f"Test MSE: {mse.item()}")
    logger.info(f"Test RMSE: {rmse.item()}")
    logger.info(f"Test Pearson correlation coefficient (PCC): {pcc}, PCC p-value: {pcc_pvalue}")
    logger.info(f"Test Kendall-Tau: {tau}, Kendall-Tau p-value: {tau_pvalue}")


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

    # Load casf as test set
    casf_cfg_path = os.path.join(get_original_cwd(), "conf", "conf_pdb", "dataset", f"{cfg.dataset.casf_dataset}.yaml")
    casf_config: DictConfig = OmegaConf.load(casf_cfg_path)
    test_dataset = PDBBindCC(
        index=casf_config.index,
        root=f"data/pdbbind/{cfg.dataset.casf_dataset}",
        lifters=list(casf_config.lifters),
        neighbor_types=list(casf_config.neighbor_types),
        connectivity=casf_config.connectivity,
        supercell=casf_config.supercell,
        connect_cross=casf_config.connect_cross,
        r_cut=casf_config.r_cut,
        force_reload=casf_config.force_reload if 'force_reload' in casf_config else False,
    )

    logger.info("Dataset loaded!")

    # ==== Get model =====
    logger.info("Loading Model...")
    model = utils.get_model(cfg, train_dataset)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    #logger.info(model)
    logger.info("Model loaded!")

    valid_dataloader = DataLoader(
        test_dataset,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # === Configure checkpoint and wandb logging ===
    ckpt_filename = f"{cfg.experiment_name}__{cfg.dataset_name}.pth"
    if cfg.ckpt_prefix is not None:
        ckpt_filename = f"{cfg.ckpt_prefix}_{ckpt_filename}"
    checkpoint_path = f"casf_val/{cfg.ckpt_dir}/{ckpt_filename}"
    logging.info(f"Checkpoint path set as: {checkpoint_path}")

    best_model = load_best_checkpoint(checkpoint_path, model)

    # ==== If eval only, evaluate and exit ====
    if cfg.eval_only:
        logger.info("Running evaluation only with best model")
        evaluate(cfg, best_model, valid_dataloader, device, mad, mean)
        return
    # ==== otherwise continue training ====




if __name__ == "__main__":
    main()
