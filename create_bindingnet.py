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
        force_reload=False,
    )
    logger.info(f"Lifted BindingNet dataset generated and stored in '{dataset.root}'.")


if __name__ == "__main__":
    main()
