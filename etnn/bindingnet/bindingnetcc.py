# General imports
from typing import Callable, List, Optional
import os
import logging
import pandas as pd
from tqdm import tqdm

# Deep Learning imports
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch

# Class imports
from etnn.lifter import Lifter, get_adjacency_types, CombinatorialComplexTransform
from etnn.bindingnet.lifts.registry import LIFTER_REGISTRY
from preprocess.single_graph_processing import create_graphs_from_dataset

logger = logging.getLogger(__name__)


class BindingNetCC(InMemoryDataset):
    def __init__(
        self,
        index: str,
        root: str,
        lifters: list[str],
        neighbor_types: list[str],
        # dim,
        connectivity: str,
        connect_cross: bool = False,
        r_cut: float = 5.0,
        merge_graphs: bool = False,
        preprocessed_graphs_path: str = "data/bindingnetcc/subset_20p_base_graphs/preprocessed",
        # merge_neighbors: str,
        supercell: Optional[bool] = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        **lifter_kwargs,
    ) -> None:
        self.index = index
        self.merge_graphs = merge_graphs
        self.preprocessed_graphs_path = preprocessed_graphs_path
        self.lifters = lifters
        self.neighbor_types = neighbor_types
        self.connectivity = connectivity
        self.connect_cross = connect_cross
        self.r_cut = r_cut if connect_cross else 0.0
        self.supercell = supercell
        self.force_reload = force_reload
        self.dim = len(lifters) - 1

        # Update dimension and lifters if supercell
        if supercell:
            self.dim += 1
            self.lifters.append("supercell:" + str(self.dim))

        # Get lifter and adjacencies
        self.lifter = Lifter(self.lifters, LIFTER_REGISTRY, self.dim, **lifter_kwargs)
        self.adjacencies = get_adjacency_types(
            self.dim,
            connectivity,
            neighbor_types,
            # visible_dims,
        )

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        # self.root e.g: data/bindingnetcc/subset_20p_crossconnect_rcut3
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return ["merged_lifted.pt"]

    def process(self) -> None:
        logger.info(f"Starting BindingNetCC processing for {self.root}")
        logger.info(f"Index file: {self.index}")
        logger.info(f"Merge graphs enabled: {self.merge_graphs}")
        logger.info(f"Connect cross: {self.connect_cross}, R_cut: {self.r_cut}")
        logger.info(f"Force reload: {self.force_reload}")
        
        df = pd.read_csv(self.index)
        logger.info(f"Loaded index with {len(df)} entries")

        merged_list = []

        lift = CombinatorialComplexTransform(
            lifter=self.lifter,
            adjacencies=self.adjacencies,
        )
        logger.info(f"Created CombinatorialComplexTransform with lifters: {self.lifters}")

        # Path for merged data
        #if self.supercell:
        #    supercell_str = 'supercell'
        #else:
        #    supercell_str = 'no_supercell'
        #if self.connect_cross:
        #    r_cut_str = f'r_cut_{self.r_cut}'
        #    connect_cross_str = 'connect_cross_' + r_cut_str
        #else:
        #    connect_cross_str = 'no_connect_cross'
        #dataset_modifications = f'{supercell_str}_{connect_cross_str}_{self.connectivity}'
        merged_data_path_root = os.path.join(self.root, f'preprocessed/merged')
        logger.info(f"Merged data path: {merged_data_path_root}")

        if self.merge_graphs:
            logger.info("Creating merged graphs from existing ligand and protein graphs")
            os.makedirs(merged_data_path_root, exist_ok=True)
            create_graphs_from_dataset(
                df, 
                self.preprocessed_graphs_path, 
                merged_data_path_root,
                self.connect_cross,
                self.r_cut,
                force_reload=self.force_reload
            ) # creates graphs and stores them in merged_data_path_root

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing BindingNetCC"):
            tuple_id = row['Target ChEMBLID'] + '_' + row['Molecule ChEMBLID']
            merged_data_path = os.path.join(merged_data_path_root, f'{tuple_id}.pt')
            if not os.path.exists(merged_data_path):
                logger.warning(f"Merged data not found for {tuple_id}")
                continue

            # Explicitly disable weights_only to load full PyG Data objects on PyTorch >=2.6
            merged_data = torch.load(merged_data_path, weights_only=False)
            merged_data = lift(merged_data)
            merged_data.id = tuple_id # keep identifier for later evaluation/logging

            y_val = torch.tensor([float(row['-logAffi'])], dtype=torch.float32)
            merged_data.y = y_val

            if (self.pre_filter is not None and not self.pre_filter(merged_data)):
                continue
            if self.pre_transform is not None:
                merged_data = self.pre_transform(merged_data)

            merged_list.append(merged_data)

        self.save(merged_list, self.processed_paths[0])
        logger.info(f"saved {len(merged_list)} graphs to {self.processed_paths[0]}")
        # self.processed_paths[0] e.g: data/bindingnetcc/subset_20p_crossconnect_rcut3/processed/merged_lifted.pt




