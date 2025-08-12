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
        # merge_neighbors: str,
        supercell: Optional[bool] = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        **lifter_kwargs,
    ) -> None:
        self.index = index
        self.lifters = lifters
        self.neighbor_types = neighbor_types
        self.connectivity = connectivity
        self.connect_cross = connect_cross
        self.r_cut = r_cut
        self.supercell = supercell
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
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return ["merged_lifted.pt"]

    def process(self) -> None:
        df = pd.read_csv(self.index)

        merged_list = []

        lift = CombinatorialComplexTransform(
            lifter=self.lifter,
            adjacencies=self.adjacencies,
        )

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing BindingNetCC"):
            tuple_id = row['Target ChEMBLID'] + '_' + row['Molecule ChEMBLID']
            merged_data_path = os.path.join(self.root, 'preprocessed/merged', f'{tuple_id}.pt')
            if not os.path.exists(merged_data_path):
                logger.warning(f"Merged data not found for {tuple_id}")
                continue

            merged_data = torch.load(merged_data_path)
            merged_data = lift(merged_data)
            #merged_data.id = tuple_id # keep identifier for later evaluation/logging

            y_val = torch.tensor([float(row['-logAffi'])], dtype=torch.float32)
            merged_data.y = y_val

            if (self.pre_filter is not None and not self.pre_filter(merged_data)):
                continue
            if self.pre_transform is not None:
                merged_data = self.pre_transform(merged_data)

            merged_list.append(merged_data)

        self.save(merged_list, self.processed_paths[0])




