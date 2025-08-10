# General imports
from typing import Callable, List, Optional
import pandas as pd
from tqdm import tqdm

# Deep Learning imports
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch

# Class imports
from etnn.lifter import Lifter, get_adjacency_types, CombinatorialComplexTransform
from etnn.bindingnet.lifts.registry import LIFTER_REGISTRY
from preprocess.single_graph_processing import (
    process_ligand_sdf, 
    process_protein_pdb_ligand_style, 
    merge_ligand_and_protein
)


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
        mode: str = 'merged',
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
        idx = {'ligand': 0, 'protein': 1, 'merged': 2}[mode]
        self.load(self.processed_paths[idx])
        # Reload the file that matches the requested mode (default: merged)
        #loaded = torch.load(self.processed_paths[idx])
        # `torch.load` may return (data, slices) or (data, slices, *extras*) depending
        # on the PyG version. We only need the first two elements.
        #self.data, self.slices = loaded[:2]

    @property
    def processed_file_names(self) -> list[str]:
        return ["ligand.pt", "protein.pt", "merged.pt"]

    def process(self) -> None:
        df = pd.read_csv(self.index)

        ligand_list, protein_list, merged_list = [], [], []

        lift = CombinatorialComplexTransform(
            lifter=self.lifter,
            adjacencies=self.adjacencies,
        )

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing BindingNetCC"):
            tuple_id = row['Target ChEMBLID'] + '_' + row['Ligand ChEMBLID']
            ligand_sdf_path = str(row['ligand_sdf_path'])
            pocket_pdb_path = str(row['pocket_pdb_path'])

            ligand_data = process_ligand_sdf(ligand_sdf_path) 
            protein_data = process_protein_pdb_ligand_style(pocket_pdb_path) 
            merged_data = merge_ligand_and_protein(
                ligand_data, 
                protein_data, 
                connect_cross=self.connect_cross,
                r_cut=self.r_cut
            )

            ligand_data.id = tuple_id
            protein_data.id = tuple_id
            merged_data.id = tuple_id

            ligand_data = lift(ligand_data)
            protein_data = lift(protein_data)
            merged_data = lift(merged_data)

            if (self.pre_filter is not None and not self.pre_filter(ligand_data)) or \
                (self.pre_filter is not None and not self.pre_filter(protein_data)) or \
                (self.pre_filter is not None and not self.pre_filter(merged_data)):
                continue
            if self.pre_transform is not None:
                ligand_data = self.pre_transform(ligand_data)
                protein_data = self.pre_transform(protein_data)
                merged_data = self.pre_transform(merged_data)

            ligand_list.append(ligand_data)
            protein_list.append(protein_data)
            merged_list.append(merged_data)

        for data_list, path in zip(
            (ligand_list, protein_list, merged_list),
            self.processed_paths
        ):
            self.save(data_list, path)




