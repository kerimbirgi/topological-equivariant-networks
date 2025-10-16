import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from etnn.combinatorial_data import Cell
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from typing import Set, List, Dict, Tuple

NUM_FEATURES = 8  # Combined features: 4 from rings + 4 from protein pockets

# Amino acid properties (simplified)
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR'}
CHARGED_POSITIVE = {'LYS', 'ARG', 'HIS'}
CHARGED_NEGATIVE = {'ASP', 'GLU'}
AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP', 'HIS'}

def combined_higher_order_lift(graph: Data, **kwargs) -> set[Cell]:
    """
    Combined lifter that extracts both ligand rings and protein pocket features.
    This avoids the data format corruption that occurs with multiple separate lifters.
    """
    cells = set()
    
    if not hasattr(graph, 'origin_nodes') or graph.origin_nodes is None:
        return cells
    
    # Extract ligand rings
    ligand_rings = extract_ligand_rings(graph, **kwargs)
    cells.update(ligand_rings)
    
    # Extract protein pocket features
    protein_features = extract_protein_pocket_features(graph, **kwargs)
    cells.update(protein_features)
    
    return cells

def extract_ligand_rings(graph: Data, **kwargs) -> set[Cell]:
    """Extract ligand rings with ring-specific features."""
    cells = set()
    
    ligand_mask = (graph.origin_nodes == 0)
    ligand_indices = torch.nonzero(ligand_mask).flatten().tolist()
    
    if not hasattr(graph, 'ligand_mol') or graph.ligand_mol is None:
        return cells
    
    mol = graph.ligand_mol
    
    try:
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            # Map local ligand indices to global graph indices
            global_ring_indices = [ligand_indices[i] for i in ring if i < len(ligand_indices)]
            
            if len(global_ring_indices) >= 3:
                # Ring features: [ring_size, is_aromatic, has_heteroatom, is_saturated]
                ring_features = compute_ring_features(ring, mol)
                # Pad with zeros for protein pocket features
                combined_features = ring_features + [0.0, 0.0, 0.0, 0.0]
                cells.add((frozenset(global_ring_indices), tuple(combined_features)))
    except Exception as e:
        print(f"Warning: Error extracting ligand rings: {e}")
    
    # Filter out rings that contain simpler rings within themselves
    filtered_cells = {
        cell
        for cell in cells
        if not any(cell[0] > other_cell[0] for other_cell in cells if cell != other_cell)
    }
    return filtered_cells

def extract_protein_pocket_features(graph: Data, **kwargs) -> set[Cell]:
    """Extract protein pocket features with protein-specific features."""
    cells = set()
    
    protein_mask = (graph.origin_nodes == 1)
    protein_indices = torch.nonzero(protein_mask).flatten().tolist()
    
    if len(protein_indices) < 3:
        return cells
    
    cluster_distance = kwargs.get('cluster_distance', 6.0)
    min_cluster_size = kwargs.get('min_cluster_size', 3)
    max_cluster_size = kwargs.get('max_cluster_size', 20)  # Limit cluster size for efficiency
    
    try:
        protein_positions = graph.pos[protein_indices].numpy()
        residue_info = extract_residue_info(graph, protein_indices)
        
        # Find different types of clusters
        hydrophobic_clusters = find_hydrophobic_clusters(
            protein_indices, protein_positions, residue_info, 
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(hydrophobic_clusters)
        
        polar_clusters = find_polar_clusters(
            protein_indices, protein_positions, residue_info, graph,
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(polar_clusters)
        
        aromatic_clusters = find_aromatic_clusters(
            protein_indices, protein_positions, residue_info, 
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(aromatic_clusters)
        
        charged_clusters = find_charged_clusters(
            protein_indices, protein_positions, residue_info, 
            cluster_distance, min_cluster_size, max_cluster_size
        )
        cells.update(charged_clusters)
        
    except Exception as e:
        print(f"Warning: Failed to extract protein pocket features: {e}")
    
    return cells

def compute_ring_features(ring_indices: Tuple[int, ...], molecule: Chem.Mol) -> List[float]:
    """Compute ring-specific features."""
    ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring_indices]
    ring_size = float(len(ring_indices))
    is_aromatic = float(all(atom.GetIsAromatic() for atom in ring_atoms))
    has_heteroatom = float(any(atom.GetSymbol() not in ("C", "H") for atom in ring_atoms))
    is_saturated = float(all(atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in ring_atoms))
    return [ring_size, is_aromatic, has_heteroatom, is_saturated]

def extract_residue_info(graph: Data, protein_indices: List[int]) -> Dict[int, Tuple[str, int]]:
    """Extract residue information from graph features."""
    residue_info = {}
    
    if not hasattr(graph, 'x') or graph.x is None:
        return residue_info
    
    # Assuming protein features have residue type info
    # This is simplified - real implementation would parse from PDB data
    for i, global_idx in enumerate(protein_indices):
        # Dummy residue info for testing
        residue_info[global_idx] = ('ALA', i // 5)  # Assign dummy residue name and ID
    return residue_info

def find_hydrophobic_clusters(protein_indices: List[int], positions: np.ndarray, 
                            residue_info: Dict[int, Tuple[str, int]], 
                            cluster_distance: float, min_size: int, max_size: int) -> Set[Cell]:
    """Find clusters of hydrophobic residues."""
    cells = set()
    hydrophobic_atom_indices = [idx for idx in protein_indices if residue_info.get(idx, ('',0))[0] in HYDROPHOBIC_RESIDUES]
    if len(hydrophobic_atom_indices) < min_size: 
        return cells
    
    # Cluster hydrophobic atoms by distance
    clusters = spatial_clustering(positions[hydrophobic_atom_indices], cluster_distance, min_size, max_size)
    
    for cluster_indices in clusters:
        # Convert back to global indices
        global_indices = [protein_indices[hydrophobic_atom_indices[i]] for i in cluster_indices]
        
        # Protein pocket features: [cluster_size, interaction_strength, charge, hydrophobicity]
        protein_features = compute_protein_cluster_features(global_indices, positions[cluster_indices], graph, is_hydrophobic=True)
        # Pad with zeros for ring features
        combined_features = [0.0, 0.0, 0.0, 0.0] + protein_features
        cells.add((frozenset(global_indices), tuple(combined_features)))
    
    return cells

def find_polar_clusters(protein_indices: List[int], positions: np.ndarray,
                       residue_info: Dict[int, Tuple[str, int]], graph: Data, 
                       cluster_distance: float, min_size: int, max_size: int) -> Set[Cell]:
    """Find clusters of polar/H-bonding atoms."""
    cells = set()
    polar_atom_indices = [idx for idx in protein_indices if residue_info.get(idx, ('',0))[0] in POLAR_RESIDUES]
    if len(polar_atom_indices) < min_size: 
        return cells
    
    clusters = spatial_clustering(positions[polar_atom_indices], cluster_distance, min_size, max_size)
    
    for cluster_indices in clusters:
        global_indices = [protein_indices[polar_atom_indices[i]] for i in cluster_indices]
        protein_features = compute_protein_cluster_features(global_indices, positions[cluster_indices], graph, is_hydrophobic=False)
        combined_features = [0.0, 0.0, 0.0, 0.0] + protein_features
        cells.add((frozenset(global_indices), tuple(combined_features)))
    
    return cells

def find_aromatic_clusters(protein_indices: List[int], positions: np.ndarray,
                          residue_info: Dict[int, Tuple[str, int]], 
                          cluster_distance: float, min_size: int, max_size: int) -> Set[Cell]:
    """Find clusters of aromatic residues (potential pi-stacking)."""
    cells = set()
    aromatic_atom_indices = [idx for idx in protein_indices if residue_info.get(idx, ('',0))[0] in AROMATIC_RESIDUES]
    if len(aromatic_atom_indices) < min_size: 
        return cells
    
    clusters = spatial_clustering(positions[aromatic_atom_indices], cluster_distance, min_size, max_size)
    
    for cluster_indices in clusters:
        global_indices = [protein_indices[aromatic_atom_indices[i]] for i in cluster_indices]
        protein_features = compute_protein_cluster_features(global_indices, positions[cluster_indices], graph, is_hydrophobic=False)
        combined_features = [0.0, 0.0, 0.0, 0.0] + protein_features
        cells.add((frozenset(global_indices), tuple(combined_features)))
    
    return cells

def find_charged_clusters(protein_indices: List[int], positions: np.ndarray,
                         residue_info: Dict[int, Tuple[str, int]], 
                         cluster_distance: float, min_size: int, max_size: int) -> Set[Cell]:
    """Find clusters of charged residues (salt bridge networks)."""
    cells = set()
    charged_atom_indices = [idx for idx in protein_indices if residue_info.get(idx, ('',0))[0] in (CHARGED_POSITIVE | CHARGED_NEGATIVE)]
    if len(charged_atom_indices) < min_size: 
        return cells
    
    clusters = spatial_clustering(positions[charged_atom_indices], cluster_distance, min_size, max_size)
    
    for cluster_indices in clusters:
        global_indices = [protein_indices[charged_atom_indices[i]] for i in cluster_indices]
        protein_features = compute_protein_cluster_features(global_indices, positions[cluster_indices], graph, is_charged=True)
        combined_features = [0.0, 0.0, 0.0, 0.0] + protein_features
        cells.add((frozenset(global_indices), tuple(combined_features)))
    
    return cells

def spatial_clustering(positions: np.ndarray, max_dist: float, min_size: int, max_size: int) -> list:
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

def compute_protein_cluster_features(cluster_atoms: List[int], protein_positions: np.ndarray, 
                                   graph: Data, is_hydrophobic: bool = False, 
                                   is_charged: bool = False) -> List[float]:
    """Compute features for protein clusters."""
    features = [0.0] * 4
    cluster_size = len(cluster_atoms)
    features[0] = float(cluster_size)
    
    if cluster_size > 1:
        positions = graph.pos[cluster_atoms]
        center = positions.mean(dim=0)
        distances = torch.norm(positions - center, dim=1)
        features[1] = 1.0 / (1.0 + distances.std().item())  # Interaction strength/compactness
    
    features[2] = 1.0 if is_charged else 0.0  # Simplified charge
    features[3] = 1.0 if is_hydrophobic else 0.0  # Simplified hydrophobicity
    
    return features

# Set the number of features as an attribute for the lifter registry
combined_higher_order_lift.num_features = NUM_FEATURES
