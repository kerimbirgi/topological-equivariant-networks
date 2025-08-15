import logging

import hydra
from omegaconf import DictConfig

from etnn.bindingnet.bindingnetcc import BindingNetCC

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/conf_bindingnet", config_name="config", version_base=None)
def main(cfg: DictConfig):
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
        merge_graphs=cfg.dataset.merge_graphs if 'merge_graphs' in cfg.dataset else False
    )
    logger.info(f"Lifted BindingNet dataset generated and stored in '{dataset.root}'.")


if __name__ == "__main__":
    main()
