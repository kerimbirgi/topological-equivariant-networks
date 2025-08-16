import torch
from torch_geometric.data import Data

from etnn.combinatorial_data import Cell

# Number of features for the supercell (global molecular complex features)
NUM_FEATURES = 5


def supercell_lift(graph: Data) -> set[Cell]:
    """
    Return the entire protein-ligand complex as a single supercell with meaningful features.
    
    This BindingNet-specific supercell lifter creates a single top-dimensional cell containing
    all nodes in the protein-ligand complex, with features that capture global properties
    of the molecular interaction.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.
        Must contain:
        - x: node features
        - origin_nodes: tensor indicating ligand (0) vs protein (1) nodes
        - pos: 3D coordinates (optional, for geometric features)

    Returns
    -------
    set[Cell]
        A singleton set containing a frozenset of all node indices and a feature
        vector with global molecular complex properties.

    Features
    --------
    The supercell features include:
    1. Number of ligand atoms
    2. Number of protein atoms  
    3. Total number of nodes
    4. Ligand-protein size ratio
    5. Complex compactness (optional, if pos available)
    """

    if (not hasattr(graph, "x")) or (graph.x is None):
        raise ValueError("The given graph does not have a feature matrix 'x'!")
    
    num_nodes = graph.x.size(0)
    if num_nodes < 2:
        return set()
    
    # Extract basic counts
    if hasattr(graph, "origin_nodes") and graph.origin_nodes is not None:
        # Count ligand vs protein nodes
        origin_nodes = graph.origin_nodes
        num_ligand = int(torch.sum(origin_nodes == 0).item())
        num_protein = int(torch.sum(origin_nodes == 1).item())
        
        # Compute size ratio (ligand/protein)
        if num_protein > 0:
            size_ratio = float(num_ligand / num_protein)
        else:
            size_ratio = float(num_ligand)  # edge case
    else:
        # Fallback: assume roughly equal split
        num_ligand = num_nodes // 2
        num_protein = num_nodes - num_ligand
        size_ratio = 1.0
    
    # Compute geometric compactness if positions available
    if hasattr(graph, "pos") and graph.pos is not None:
        pos = graph.pos
        # Simple compactness: inverse of average pairwise distance
        distances = torch.cdist(pos, pos, p=2)
        # Exclude diagonal (self-distances = 0)
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=distances.device)
        avg_distance = torch.mean(distances[mask])
        compactness = 1.0 / (1.0 + avg_distance.item())  # normalized compactness
    else:
        compactness = 0.5  # neutral value when positions unavailable
    
    # Create feature vector
    features = (
        float(num_ligand),          # Feature 0: Number of ligand atoms
        float(num_protein),         # Feature 1: Number of protein atoms  
        float(num_nodes),           # Feature 2: Total number of nodes
        size_ratio,                 # Feature 3: Ligand-protein size ratio
        compactness,                # Feature 4: Complex compactness
    )
    
    # Create supercell containing all nodes
    all_nodes = frozenset(range(num_nodes))
    
    return {(all_nodes, features)}


def get_supercell_num_features(graph: Data) -> int:
    """Determine number of features for BindingNet supercell"""
    return NUM_FEATURES


# Set the num_features attribute for compatibility
supercell_lift.num_features = NUM_FEATURES
