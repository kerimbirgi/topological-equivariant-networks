import logging
import os
from datetime import datetime

import hydra
from omegaconf import DictConfig

from etnn.bindingnet.bindingnetcc import BindingNetCC
from preprocess.single_graph_processing_pdb import create_single_graphs

@hydra.main(config_path="conf/conf_bindingnet", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup simple logging with file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("create_bindingnet.log", mode="w"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting BindingNet dataset creation")
    logger.info(f"Dataset: {cfg.dataset}")
    logger.info(f"Create single graphs: {cfg.dataset.create_single_graphs}")
    logger.info(f"Merge graphs: {cfg.dataset.merge_graphs}")
    logger.info(f"Force reload: {cfg.dataset.force_reload}")
    logger.info(f"Single graphs path: {cfg.dataset.single_graphs_path}")

    logger.info(f"Supercell: {cfg.dataset.supercell}")
    logger.info(f"Connectivity: {cfg.dataset.connectivity}")
    logger.info(f"Neighbors: {cfg.dataset.neighbor_types}")

    if 'single_graphs_path' in cfg.dataset and cfg.dataset.create_single_graphs:
        create_single_graphs(cfg.dataset.index, cfg.dataset.single_graphs_path)
    
    dataset = BindingNetCC(
        index=cfg.dataset.index,
        root=f"data/bindingnetcc/{cfg.dataset_name}",
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        supercell=cfg.dataset.supercell,
        connect_cross=cfg.dataset.connect_cross,
        r_cut=cfg.dataset.r_cut,
        force_reload=cfg.dataset.force_reload if 'force_reload' in cfg.dataset else False,
        merge_graphs=cfg.dataset.merge_graphs if 'merge_graphs' in cfg.dataset else False,
        single_graphs_path=cfg.dataset.single_graphs_path if 'single_graphs_path' in cfg.dataset else "data/bindingnetcc/base_graphs_rbf/preprocessed",
        merged_graphs_path=cfg.dataset.merged_graphs_path if "merged_graphs_path" in cfg.dataset else "data/bindingnetcc/base_graphs_rbf/preprocessed",
    )
    
    logger.info(f"Dataset created successfully! Length: {len(dataset)}")
    logger.info(f"Lifted BindingNet dataset generated and stored in '{dataset.root}'.")


if __name__ == "__main__":
    main()
