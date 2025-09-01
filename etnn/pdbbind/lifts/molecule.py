"""
Heterogeneous lifter for rank 3: Ligand Molecule + Protein Molecule
"""
import torch
from torch_geometric.data import Data
from etnn.combinatorial_data import Cell

# Feature dimensions must match for both molecules
NUM_FEATURES = 8

def molecule_lift(graph: Data, **kwargs) -> set[Cell]:
    """
    Create rank 3 cells containing ligand molecule and protein molecule.
    
    Returns
    -------
    set[Cell]
        Set containing exactly 2 cells: ligand molecule and protein molecule
    """
    cells = set()
    
    # Check required attributes
    if not hasattr(graph, 'origin_nodes') or graph.origin_nodes is None:
        print("Warning: Graph missing origin_nodes, cannot distinguish ligand/protein")
        return cells
    
    # Get ligand and protein atom indices
    origin_nodes = graph.origin_nodes
    ligand_mask = (origin_nodes == 0)
    protein_mask = (origin_nodes == 1)
    
    ligand_atoms = torch.nonzero(ligand_mask).flatten().tolist()
    protein_atoms = torch.nonzero(protein_mask).flatten().tolist()
    
    # Create ligand molecule cell
    if len(ligand_atoms) > 0:
        ligand_features = compute_ligand_molecule_features(ligand_atoms, graph)
        cells.add((frozenset(ligand_atoms), tuple(ligand_features)))
    
    # Create protein molecule cell  
    if len(protein_atoms) > 0:
        protein_features = compute_protein_molecule_features(protein_atoms, graph)
        cells.add((frozenset(protein_atoms), tuple(protein_features)))
    
    return cells

def compute_ligand_molecule_features(ligand_atoms: list, graph: Data) -> list:
    """Compute molecular-level features for the ligand."""
    features = [0.0] * NUM_FEATURES
    
    try:
        # Feature 1: Number of atoms
        features[0] = len(ligand_atoms)
        
        # Feature 2: Molecular weight (approximated)
        if hasattr(graph, 'x') and graph.x is not None:
            atom_features = graph.x[ligand_atoms]
            # Assuming first feature is atomic number
            if atom_features.shape[1] > 0:
                atomic_nums = atom_features[:, 0]
                # Simplified molecular weight calculation
                weight_map = {1: 1.0, 6: 12.0, 7: 14.0, 8: 16.0, 9: 19.0, 15: 31.0, 16: 32.0, 17: 35.5}
                weights = [weight_map.get(int(num), 12.0) for num in atomic_nums]
                features[1] = sum(weights)
        
        # Feature 3: Spatial extent (molecular size)
        if len(ligand_atoms) >= 2:
            positions = graph.pos[ligand_atoms]
            extent = (positions.max(dim=0)[0] - positions.min(dim=0)[0]).norm()
            features[2] = float(extent)
        
        # Feature 4: Molecular polarity (approximated by charge distribution)
        if hasattr(graph, 'x') and graph.x is not None:
            atom_features = graph.x[ligand_atoms]
            if atom_features.shape[1] > 2:  # Assuming 3rd feature is formal charge
                charges = atom_features[:, 2]
                features[3] = float(charges.abs().sum())
        
        # Feature 5: Molecule type indicator
        features[4] = 1.0  # 1 = ligand
        
        # Feature 6: Aromatic atom ratio
        if hasattr(graph, 'x') and graph.x is not None:
            atom_features = graph.x[ligand_atoms]
            if atom_features.shape[1] > 5:  # Assuming 6th feature is aromaticity
                aromatic_count = atom_features[:, 5].sum()
                features[5] = float(aromatic_count / len(ligand_atoms))
        
        # Features 7-8: Reserved for additional molecular descriptors
        features[6] = 0.0
        features[7] = 0.0
        
    except Exception as e:
        print(f"Warning: Error computing ligand molecule features: {e}")
        features = [0.0] * NUM_FEATURES
    
    return features

def compute_protein_molecule_features(protein_atoms: list, graph: Data) -> list:
    """Compute molecular-level features for the protein."""
    features = [0.0] * NUM_FEATURES
    
    try:
        # Feature 1: Number of atoms  
        features[0] = len(protein_atoms)
        
        # Feature 2: Molecular weight (approximated)
        if hasattr(graph, 'x') and graph.x is not None:
            atom_features = graph.x[protein_atoms]
            if atom_features.shape[1] > 0:
                atomic_nums = atom_features[:, 0]
                weight_map = {1: 1.0, 6: 12.0, 7: 14.0, 8: 16.0, 9: 19.0, 15: 31.0, 16: 32.0, 17: 35.5}
                weights = [weight_map.get(int(num), 12.0) for num in atomic_nums]
                features[1] = sum(weights)
        
        # Feature 3: Spatial extent (protein pocket size)
        if len(protein_atoms) >= 2:
            positions = graph.pos[protein_atoms]
            extent = (positions.max(dim=0)[0] - positions.min(dim=0)[0]).norm()
            features[2] = float(extent)
        
        # Feature 4: Charge distribution
        if hasattr(graph, 'x') and graph.x is not None:
            atom_features = graph.x[protein_atoms]
            if atom_features.shape[1] > 2:
                charges = atom_features[:, 2]
                features[3] = float(charges.abs().sum())
        
        # Feature 5: Molecule type indicator
        features[4] = 0.0  # 0 = protein
        
        # Feature 6: Aromatic atom ratio
        if hasattr(graph, 'x') and graph.x is not None:
            atom_features = graph.x[protein_atoms]
            if atom_features.shape[1] > 5:
                aromatic_count = atom_features[:, 5].sum()
                features[5] = float(aromatic_count / len(protein_atoms))
        
        # Features 7-8: Reserved
        features[6] = 0.0
        features[7] = 0.0
        
    except Exception as e:
        print(f"Warning: Error computing protein molecule features: {e}")
        features = [0.0] * NUM_FEATURES
    
    return features

# Set feature count for compatibility
molecule_lift.num_features = NUM_FEATURES
