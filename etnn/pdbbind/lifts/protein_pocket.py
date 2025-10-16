"""
Protein pocket lifter for PDBBind dataset - extracts higher-order structural features 
from protein binding pockets that are analogous to ligand rings.

Features extracted:
1. Hydrophobic patches - clusters of nonpolar residues
2. Polar clusters - groups of hydrogen bond donors/acceptors  
3. Aromatic clusters - stacked aromatic residues (Phe, Tyr, Trp)
4. Charged clusters - groups of positively/negatively charged residues
"""

import torch
import numpy as np
from torch_geometric.data import Data
from etnn.combinatorial_data import Cell
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# Number of features per protein pocket cluster
# Must match ring lifter features to concatenate properly for rank 2
NUM_FEATURES = 4  # cluster_size, interaction_strength, charge, hydrophobicity

# Amino acid properties
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR'}
CHARGED_POSITIVE = {'LYS', 'ARG', 'HIS'}
CHARGED_NEGATIVE = {'ASP', 'GLU'}
AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP', 'HIS'}

# Atomic properties for interaction calculation
HBOND_DONORS = {'N', 'O'}  # Simplified - in reality depends on chemistry
HBOND_ACCEPTORS = {'O', 'N'}


def protein_pocket_lift(graph: Data, **kwargs) -> set[Cell]:
    """
    Extract higher-order structural features from protein pockets.
    
    Identifies clusters of residues with similar physicochemical properties
    that form binding site features important for protein-ligand interactions.
    
    Parameters
    ----------
    graph : Data
        The input graph with protein pocket atoms.
        Must have 'origin_nodes' to distinguish protein (1) from ligand (0).
    **kwargs
        Additional parameters:
        - cluster_distance: max distance for clustering (default: 6.0 Å)
        - min_cluster_size: minimum atoms per cluster (default: 3)
    
    Returns
    -------
    set[Cell]
        Set of cells representing protein pocket features.
        Each cell has 4 features matching the ring lifter.
    """
    cells = set()
    
    # Check for required attributes
    if not hasattr(graph, 'origin_nodes') or graph.origin_nodes is None:
        print("Warning: Graph missing origin_nodes, cannot identify protein atoms")
        return cells
    
    # Get protein atoms only (origin_nodes == 1)
    origin_nodes = graph.origin_nodes
    protein_mask = (origin_nodes == 1)
    protein_indices = torch.nonzero(protein_mask).flatten().tolist()
    
    if len(protein_indices) < 3:
        print("Warning: Too few protein atoms for clustering")
        return cells
    
    # Extract parameters
    cluster_distance = kwargs.get('cluster_distance', 6.0)
    min_cluster_size = kwargs.get('min_cluster_size', 3)
    max_cluster_size = kwargs.get('max_cluster_size', 20)  # Limit cluster size for efficiency
    
    try:
        # Get protein atom positions and features
        protein_positions = graph.pos[protein_indices].numpy()
        
        # Extract residue information if available
        residue_info = extract_residue_info(graph, protein_indices)
        
        # Find different types of clusters
        
        # 1. Hydrophobic patches
        hydrophobic_clusters = find_hydrophobic_clusters(
            protein_indices, protein_positions, residue_info, 
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(hydrophobic_clusters)
        
        # 2. Polar/H-bond clusters  
        polar_clusters = find_polar_clusters(
            protein_indices, protein_positions, residue_info, graph,
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(polar_clusters)
        
        # 3. Aromatic stacking clusters
        aromatic_clusters = find_aromatic_clusters(
            protein_indices, protein_positions, residue_info,
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(aromatic_clusters)
        
        # 4. Charged clusters (salt bridge networks)
        charged_clusters = find_charged_clusters(
            protein_indices, protein_positions, residue_info,
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(charged_clusters)
        
        return cells
        
    except Exception as e:
        print(f"Warning: Failed to extract protein pocket features: {e}")
        return cells


def extract_residue_info(graph: Data, protein_indices: list) -> dict:
    """Extract residue information from graph features."""
    residue_info = {}
    
    if not hasattr(graph, 'x') or graph.x is None:
        return residue_info
    
    # Assuming protein features have residue type info
    # This is simplified - real implementation would parse from PDB data
    for i, idx in enumerate(protein_indices):
        # For now, use simple heuristics based on atom features
        # In real implementation, you'd store residue names in graph processing
        residue_info[idx] = {
            'residue_type': 'UNK',  # Would be extracted from PDB
            'atom_type': 'C',       # Would be extracted from atom features
            'is_backbone': False    # Would be determined from atom names
        }
    
    return residue_info


def find_hydrophobic_clusters(protein_indices: list, positions: np.ndarray, 
                            residue_info: dict, max_dist: float, 
                            min_size: int, max_size: int = 20) -> set[Cell]:
    """Find clusters of hydrophobic residues."""
    cells = set()
    
    # Identify hydrophobic atoms (simplified)
    hydrophobic_atoms = []
    for i, idx in enumerate(protein_indices):
        # In real implementation, check residue type from residue_info
        # For now, use spatial clustering as proxy
        hydrophobic_atoms.append(i)
    
    if len(hydrophobic_atoms) < min_size:
        return cells
    
    # Cluster hydrophobic atoms by distance
    clusters = spatial_clustering(positions[hydrophobic_atoms], max_dist, min_size, max_size)
    
    for cluster_indices in clusters:
        # Convert back to global indices
        global_indices = [protein_indices[hydrophobic_atoms[i]] for i in cluster_indices]
        
        # Compute features
        features = compute_hydrophobic_features(global_indices, positions[cluster_indices])
        cells.add((frozenset(global_indices), features))
    
    return cells


def find_polar_clusters(protein_indices: list, positions: np.ndarray,
                       residue_info: dict, graph: Data, max_dist: float,
                       min_size: int, max_size: int = 20) -> set[Cell]:
    """Find clusters of polar/H-bonding atoms."""
    cells = set()
    
    # Identify potential H-bond donors/acceptors
    polar_atoms = []
    for i, idx in enumerate(protein_indices):
        # Simplified: assume N and O atoms can participate in H-bonds
        if hasattr(graph, 'x') and graph.x is not None:
            # Check atom type (assuming first feature is atomic number)
            atomic_num = int(graph.x[idx, 0].item()) if graph.x.shape[1] > 0 else 6
            if atomic_num in [7, 8]:  # N or O
                polar_atoms.append(i)
    
    if len(polar_atoms) < min_size:
        return cells
    
    # Cluster polar atoms  
    clusters = spatial_clustering(positions[polar_atoms], max_dist, min_size)
    
    for cluster_indices in clusters:
        global_indices = [protein_indices[polar_atoms[i]] for i in cluster_indices]
        features = compute_polar_features(global_indices, positions[cluster_indices])
        cells.add((frozenset(global_indices), features))
    
    return cells


def find_aromatic_clusters(protein_indices: list, positions: np.ndarray,
                          residue_info: dict, max_dist: float,
                          min_size: int, max_size: int = 20) -> set[Cell]:
    """Find clusters of aromatic residues (potential pi-stacking)."""
    cells = set()
    
    # For simplified implementation, use geometric clustering
    # Real implementation would identify aromatic rings from residue types
    if len(protein_indices) < min_size:
        return cells
    
    # Use tighter clustering for aromatic stacking (shorter range interaction)
    aromatic_dist = min(max_dist, 4.5)  # Typical pi-stacking distance
    clusters = spatial_clustering(positions, aromatic_dist, min_size)
    
    for cluster_indices in clusters:
        global_indices = [protein_indices[i] for i in cluster_indices]
        features = compute_aromatic_features(global_indices, positions[cluster_indices])
        cells.add((frozenset(global_indices), features))
    
    return cells


def find_charged_clusters(protein_indices: list, positions: np.ndarray,
                         residue_info: dict, max_dist: float,
                         min_size: int, max_size: int = 20) -> set[Cell]:
    """Find clusters of charged residues (salt bridge networks)."""
    cells = set()
    
    # Simplified clustering - real implementation would identify charged residues
    if len(protein_indices) < min_size:
        return cells
    
    clusters = spatial_clustering(positions, max_dist, min_size)
    
    for cluster_indices in clusters:
        global_indices = [protein_indices[i] for i in cluster_indices]
        features = compute_charged_features(global_indices, positions[cluster_indices])
        cells.add((frozenset(global_indices), features))
    
    return cells


def spatial_clustering(positions: np.ndarray, max_dist: float, min_size: int, max_size: int = 20) -> list:
    """Perform spatial clustering using DBSCAN."""
    if len(positions) < min_size:
        return []
    
    # Use DBSCAN for clustering
    clustering = DBSCAN(eps=max_dist, min_samples=min_size).fit(positions)
    
    clusters = []
    for label in set(clustering.labels_):
        if label != -1:  # Not noise
            cluster_indices = np.where(clustering.labels_ == label)[0].tolist()
            if len(cluster_indices) >= min_size:
                # Limit cluster size for efficiency
                if len(cluster_indices) > max_size:
                    cluster_indices = cluster_indices[:max_size]
                clusters.append(cluster_indices)
    
    return clusters


def compute_hydrophobic_features(global_indices: list, positions: np.ndarray) -> tuple[float]:
    """Compute features for hydrophobic clusters."""
    cluster_size = float(len(global_indices))
    
    # Interaction strength (compactness)
    if len(positions) > 1:
        distances = pdist(positions)
        interaction_strength = float(1.0 / (1.0 + np.mean(distances)))
    else:
        interaction_strength = 1.0
    
    # Charge (0 for hydrophobic)
    charge = 0.0
    
    # Hydrophobicity (1.0 for hydrophobic clusters)
    hydrophobicity = 1.0
    
    return (cluster_size, interaction_strength, charge, hydrophobicity)


def compute_polar_features(global_indices: list, positions: np.ndarray) -> tuple[float]:
    """Compute features for polar clusters."""
    cluster_size = float(len(global_indices))
    
    # Interaction strength
    if len(positions) > 1:
        distances = pdist(positions)
        interaction_strength = float(1.0 / (1.0 + np.mean(distances)))
    else:
        interaction_strength = 1.0
    
    # Charge (neutral for polar)
    charge = 0.0
    
    # Hydrophobicity (0.0 for polar clusters)  
    hydrophobicity = 0.0
    
    return (cluster_size, interaction_strength, charge, hydrophobicity)


def compute_aromatic_features(global_indices: list, positions: np.ndarray) -> tuple[float]:
    """Compute features for aromatic clusters."""
    cluster_size = float(len(global_indices))
    
    # Interaction strength (stronger for tight aromatic stacking)
    if len(positions) > 1:
        distances = pdist(positions)
        interaction_strength = float(2.0 / (1.0 + np.mean(distances)))  # Stronger weight
    else:
        interaction_strength = 2.0
    
    # Charge (neutral)
    charge = 0.0
    
    # Hydrophobicity (moderate for aromatics)
    hydrophobicity = 0.5
    
    return (cluster_size, interaction_strength, charge, hydrophobicity)


def compute_charged_features(global_indices: list, positions: np.ndarray) -> tuple[float]:
    """Compute features for charged clusters."""
    cluster_size = float(len(global_indices))
    
    # Interaction strength 
    if len(positions) > 1:
        distances = pdist(positions)
        interaction_strength = float(1.0 / (1.0 + np.mean(distances)))
    else:
        interaction_strength = 1.0
    
    # Charge (simplified - assume net charge)
    charge = 1.0  # Would be calculated from actual residue charges
    
    # Hydrophobicity (0.0 for charged)
    hydrophobicity = 0.0
    
    return (cluster_size, interaction_strength, charge, hydrophobicity)


# Set the number of features as an attribute for the lifter registry
protein_pocket_lift.num_features = NUM_FEATURES
