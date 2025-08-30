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


logger = logging.getLogger(__name__)

def load_checkpoint_eval(checkpoint_path, model):
    best_model = copy.deepcopy(model)
    device = next(model.parameters()).device
    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint!")
        checkpoint = torch.load(checkpoint_path)
        model.to("cpu")
        best_model.to("cpu")
        model.load_state_dict(checkpoint["model"])
        best_model.load_state_dict(checkpoint["best_model"])
        model.to(device)
        best_model.to(device)

        
        return model, best_model
    else:
        raise FileNotFoundError("Checkpoint does not exist")

def evaluate(cfg: DictConfig, model, test_dataloader, device, results_dir = 'casf_eval', save_eval = False):
    os.makedirs(results_dir, exist_ok=True)

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


    # Compute metrics on CPU
    mae = torch.nn.functional.l1_loss(preds, targets, reduction='mean')
    mse = torch.nn.functional.mse_loss(preds, targets, reduction='mean')
    rmse = torch.sqrt(mse)
    pcc, pcc_pvalue = pearsonr(preds, targets)
    tau, tau_pvalue = kendalltau(preds, targets, variant="b")  # (tau, p-value)
    predictions_range = f"[{torch.min(preds)}, {torch.max(preds)}]"
    targets_range = f"[{torch.min(targets)}, {torch.max(targets)}]"

    logger.info(f"Test MAE: {mae.item()}")
    logger.info(f"Test MSE: {mse.item()}")
    logger.info(f"Test RMSE: {rmse.item()}")
    logger.info(f"Test Pearson correlation coefficient (PCC): {pcc}, PCC p-value: {pcc_pvalue}")
    logger.info(f"Test Kendall-Tau: {tau}, Kendall-Tau p-value: {tau_pvalue}")
    logger.info(f"Predictions range: {predictions_range}")
    logger.info(f"Targets range: {targets_range}")


    # Save predictions/targets to CSV as plain numeric arrays
    if save_eval:
        df = pd.DataFrame({
            'predictions': preds.numpy().ravel(),
            'targets': targets.numpy().ravel(),
        })
        df.to_csv(os.path.join(results_dir, f'{cfg.experiment_name}_{cfg.dataset_name}_predictions.csv'), index=False)
        with open(os.path.join('casf_eval', f'{cfg.experiment_name}_{cfg.dataset_name}_evaluation.txt'), 'w') as f:
            f.write(f"Test MAE: {mae.item()}\n")
            f.write(f"Test MSE: {mse.item()}\n")
            f.write(f"Test RMSE: {rmse.item()}\n")
            f.write(f"Test Pearson correlation coefficient (PCC): {pcc}, PCC p-value: {pcc_pvalue}\n")
            f.write(f"Test Kendall-Tau: {tau}, Kendall-Tau p-value: {tau_pvalue}\n")
            f.write(f"Predictions range: {predictions_range}\n")
            f.write(f"Targets range: {targets_range}\n")


@hydra.main(config_path="conf/conf_bindingnet", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # ==== Initial setup =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load casf as test set
    casf_cfg_path = os.path.join(get_original_cwd(), "conf", "conf_pdb", "casf.yaml")
    casf_config: DictConfig = OmegaConf.load(casf_cfg_path)
    test_dataset = PDBBindCC(
        index=casf_config.dataset.index,
        root=f"data/pdbbind/{casf_config.dataset_name}",
        lifters=list(casf_config.dataset.lifters),
        neighbor_types=list(casf_config.dataset.neighbor_types),
        connectivity=casf_config.dataset.connectivity,
        supercell=casf_config.dataset.supercell,
        connect_cross=casf_config.dataset.connect_cross,
        r_cut=casf_config.dataset.r_cut,
        force_reload=casf_config.dataset.force_reload if 'force_reload' in casf_config.dataset else False,
    )

    logger.info("Dataset loaded!")

    # ==== Get model =====
    logger.info("Loading Model...")
    model = utils.get_model(cfg, test_dataset)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    #logger.info(model)
    logger.info("Model loaded!")

    test_dataloader = DataLoader(
        test_dataset,
        num_workers=cfg.training.num_workers if 'num_workers' in cfg.training else 4,
        pin_memory=cfg.training.pin_memory if 'pin_memory' in cfg.training else True,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # === Configure checkpoint and wandb logging ===
    ckpt_filename = f"{cfg.experiment_name}__{cfg.dataset_name}.pth"
    if cfg.ckpt_prefix is not None:
        ckpt_filename = f"{cfg.ckpt_prefix}_{ckpt_filename}"
    checkpoint_path = f"{cfg.ckpt_dir}/{ckpt_filename}"
    logging.info(f"Checkpoint path set as: {checkpoint_path}")

    model, best_model = load_checkpoint_eval(checkpoint_path, model)

    logger.info("Running evaluation with best model")
    evaluate(cfg, best_model, test_dataloader, device)



    

if __name__ == "__main__":
    main()
