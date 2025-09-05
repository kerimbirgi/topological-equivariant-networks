"""
Cross-molecular bond lifter for rank 2: Ligand-Protein Interactions

This lifter creates rank 2 cells representing intermolecular interactions between
ligand and protein atoms, including hydrogen bonds, salt bridges, π-π stacking,
cation-π interactions, and hydrophobic contacts.
"""
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, rdMolChemicalFeatures
import os
import torch
from torch_geometric.data import Data
from etnn.combinatorial_data import Cell

# Feature dimensions for cross-molecular interactions
NUM_FEATURES = 20  # 15(RBF) + 5(interaction_types)

def bond_cross_lift(graph: Data, **kwargs) -> set[Cell]:
    """
    Create rank 2 cells for intermolecular interactions between ligand and protein atoms.
    
    Parameters
    ---------- 
    graph : Data
        The merged ligand-protein graph with origin_nodes attribute
    **kwargs : dict
        Additional parameters including:
        - r_cut : float, default=5.0 - Distance cutoff for considering interactions (Angstroms)
        - M_RBF : int, default=15 - Number of RBF basis functions for distance encoding
        
    Returns
    -------
    set[Cell]
        Set of cells where each cell represents a ligand-protein interaction pair
    """
    # Extract parameters from kwargs with defaults
    r_cut = kwargs.get('lifter_r_cut', kwargs.get('r_cut', 3.0))
    M_RBF = kwargs.get('M_RBF', 15)
    
    # Debug: print received parameters (remove this line for production)
    # print(f"bond_cross_lift: r_cut={r_cut}, M_RBF={M_RBF}, received kwargs keys: {list(kwargs.keys())}")
    
    return _bond_cross_lift_impl(graph, r_cut, M_RBF)

def _bond_cross_lift_impl(graph: Data, r_cut: float = 5.0, M_RBF: int = 15) -> set[Cell]:
    """
    Implementation of cross-molecular bond lifter for rank 2: Ligand-Protein Interactions.
    
    Parameters
    ----------
    graph : Data
        The merged ligand-protein graph with origin_nodes attribute
    r_cut : float, default=5.0
        Distance cutoff for considering interactions (Angstroms)
    M_RBF : int, default=15
        Number of RBF basis functions for distance encoding
        
    Returns
    -------
    set[Cell]
        Set of cells where each cell represents a ligand-protein interaction pair
    """
    cells = set()
    
    # Check required attributes
    if not hasattr(graph, 'origin_nodes') or graph.origin_nodes is None:
        print("Warning: Graph missing origin_nodes, cannot distinguish ligand/protein")
        return cells
    
    if not hasattr(graph, 'pos') or graph.pos is None:
        print("Warning: Graph missing pos, cannot compute distances")
        return cells
        
    if not hasattr(graph, 'ligand_mol') or not hasattr(graph, 'protein_mol'):
        print("Warning: Graph missing mol objects, cannot compute pharmacophore features")
        return cells
    
    # Get ligand and protein atom indices
    origin_nodes = graph.origin_nodes
    ligand_mask = (origin_nodes == 0)
    protein_mask = (origin_nodes == 1)
    
    ligand_atoms = torch.nonzero(ligand_mask).flatten()
    protein_atoms = torch.nonzero(protein_mask).flatten()
    
    num_lig = ligand_atoms.size(0)
    num_pro = protein_atoms.size(0)
    
    if num_lig == 0 or num_pro == 0:
        return cells
    
    # Compute pairwise distances
    pos = graph.pos
    lig_pos = pos[ligand_atoms]  # [num_lig, 3]
    pro_pos = pos[protein_atoms]  # [num_pro, 3]
    
    d = torch.cdist(lig_pos, pro_pos)  # [num_lig, num_pro], pairwise euclidean distances
    
    # Find pairs within cutoff
    src, dst = torch.nonzero(d < r_cut, as_tuple=True)  # ligand idx, protein idx within their respective groups
    
    if src.size(0) == 0:
        return cells
    
    # Map to global indices
    src_global = ligand_atoms[src]  
    dst_global = protein_atoms[dst]  
    
    # Get distances for selected pairs
    distances = d[src, dst]  # [num_pairs]
    
    # Compute pharmacophore features for interaction classification
    lig_flags = _atom_flags_pharmacophore(graph.ligand_mol)
    pro_flags = _atom_flags_pharmacophore(graph.protein_mol)
    
    # Create cells for each interaction pair
    for i in range(src.size(0)):
        lig_idx = src[i].item()      # Index within ligand atoms
        pro_idx = dst[i].item()      # Index within protein atoms
        lig_global = src_global[i].item()  # Global atom index
        pro_global = dst_global[i].item()  # Global atom index
        distance = distances[i].item()
        
        # Compute interaction features for this pair
        features = compute_cross_interaction_features(
            lig_idx, pro_idx, distance, lig_flags, pro_flags, r_cut, M_RBF
        )
        
        # Create cell with both atoms as members
        cell_atoms = frozenset([lig_global, pro_global])
        cells.add((cell_atoms, tuple(features)))
    
    return cells


def compute_cross_interaction_features(
    lig_idx: int, 
    pro_idx: int, 
    distance: float,
    lig_flags: dict,
    pro_flags: dict,
    r_cut: float,
    M_RBF: int
) -> list[float]:
    """
    Compute feature vector for a ligand-protein interaction pair.
    
    Features include:
    1. RBF-encoded distance (M_RBF dimensions)  
    2. Interaction type flags (5D): [hbond, salt_bridge, pi_pi, cation_pi, hydrophobic]
    
    Parameters
    ----------
    lig_idx : int
        Index of ligand atom within ligand
    pro_idx : int  
        Index of protein atom within protein
    distance : float
        Distance between the atoms (Angstroms)
    lig_flags : dict
        Pharmacophore flags for ligand atoms
    pro_flags : dict
        Pharmacophore flags for protein atoms
    r_cut : float
        Distance cutoff used for RBF encoding
    M_RBF : int
        Number of RBF basis functions
        
    Returns
    -------
    list[float]
        Feature vector of length M_RBF + 5
    """
    features = []
    
    # 1. RBF-encoded distance
    rbf_features = rbf_expand(torch.tensor([distance]), r_cut=r_cut, M=M_RBF)
    features.extend(rbf_features.squeeze().tolist())
    
    # 2. Interaction type classification
    # Get pharmacophore properties for both atoms
    lig_donor = lig_flags["donor"][lig_idx].item()
    lig_acceptor = lig_flags["acceptor"][lig_idx].item()
    lig_cationic = lig_flags["cationic"][lig_idx].item()
    lig_anionic = lig_flags["anionic"][lig_idx].item()
    lig_aromatic = lig_flags["aromatic"][lig_idx].item()
    lig_hydrophobe = lig_flags["hydrophobe"][lig_idx].item()
    
    pro_donor = pro_flags["donor"][pro_idx].item()
    pro_acceptor = pro_flags["acceptor"][pro_idx].item()
    pro_cationic = pro_flags["cationic"][pro_idx].item()
    pro_anionic = pro_flags["anionic"][pro_idx].item()
    pro_aromatic = pro_flags["aromatic"][pro_idx].item()
    pro_hydrophobe = pro_flags["hydrophobe"][pro_idx].item()
    
    # Compute interaction flags based on pharmacophore compatibility
    # Distance filtering already handled by r_cut, let RBF features encode distance effects
    # H-bond: donor-acceptor pairs
    hbond = (lig_donor and pro_acceptor) or (lig_acceptor and pro_donor)
    # Salt bridge: oppositely charged atoms
    salt_bridge = (lig_cationic and pro_anionic) or (lig_anionic and pro_cationic)
    # π-π stacking: aromatic-aromatic atoms
    pi_pi = (lig_aromatic and pro_aromatic)
    # Cation-π: charged-aromatic atoms
    cation_pi = (lig_cationic and pro_aromatic) or (lig_aromatic and pro_cationic)
    # Hydrophobic contact: non-polar atoms
    hydrophobe_contact = (lig_hydrophobe and pro_hydrophobe)
    
    # Add interaction flags as binary features
    features.extend([
        float(hbond),
        float(salt_bridge), 
        float(pi_pi),
        float(cation_pi),
        float(hydrophobe_contact)
    ])
    
    return features


def rbf_expand(distances: torch.Tensor, r_cut: float, M: int) -> torch.Tensor:
    """
    Expand distances using radial basis functions (RBF).
    
    Parameters
    ----------
    distances : torch.Tensor
        Distances to encode, shape [N]
    r_cut : float
        Cutoff distance 
    M : int
        Number of RBF basis functions
        
    Returns
    -------
    torch.Tensor
        RBF-encoded distances, shape [N, M]
    """
    # Create RBF centers from 0 to r_cut
    centers = torch.linspace(0, r_cut, M)
    
    # RBF width parameter
    width = r_cut / M
    
    # Compute RBF features: exp(-||d - c||^2 / (2 * width^2))
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)  # [N, M]
    rbf = torch.exp(-(diff ** 2) / (2 * width ** 2))
    
    return rbf


def _atom_flags_pharmacophore(mol) -> dict:
    """
    Compute pharmacophore flags for all atoms in a molecule.
    
    This function extracts pharmacophore properties (donor, acceptor, aromatic, etc.)
    for each atom in the molecule using RDKit.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
        
    Returns
    -------
    dict
        Dictionary with keys: donor, acceptor, cationic, anionic, aromatic, hydrophobe
        Each value is a tensor of shape [num_atoms] with boolean flags
    """
    
    num_atoms = mol.GetNumAtoms()
    
    # Initialize flag arrays
    flags = {
        "donor": torch.zeros(num_atoms, dtype=torch.uint8),
        "acceptor": torch.zeros(num_atoms, dtype=torch.uint8),
        "cationic": torch.zeros(num_atoms, dtype=torch.uint8),
        "anionic": torch.zeros(num_atoms, dtype=torch.uint8),
        "aromatic": torch.zeros(num_atoms, dtype=torch.uint8),
        "hydrophobe": torch.zeros(num_atoms, dtype=torch.uint8),
    }
    
    # Try to extract pharmacophore features, but handle cases where it fails
    # (e.g., for large protein structures where pharmacophore analysis isn't suitable)
    try:
        # Ensure ring information is computed before pharmacophore analysis
        try:
            # This initializes the ring information that RDKit needs
            if rdMolDescriptors is not None:
                rdMolDescriptors.CalcNumRings(mol)
        except Exception:
            # If ring computation fails, continue without it
            pass
        
        # Load pharmacophore feature factory
        if rdMolChemicalFeatures is None:
            raise ImportError("rdMolChemicalFeatures not available")
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = rdMolChemicalFeatures.BuildFeatureFactory(fdefName)
        
        # Get features for the molecule
        num_features = factory.GetNumMolFeatures(mol)
        
        # Process each pharmacophore feature
        for i in range(num_features):
            feat = factory.GetMolFeature(mol, i)
            feat_type = feat.GetType()
            atom_indices = feat.GetAtomIds()
            
            for atom_idx in atom_indices:
                if feat_type == 'Donor':
                    flags["donor"][atom_idx] = 1
                elif feat_type == 'Acceptor':
                    flags["acceptor"][atom_idx] = 1
                elif feat_type == 'PosIonizable':
                    flags["cationic"][atom_idx] = 1
                elif feat_type == 'NegIonizable':
                    flags["anionic"][atom_idx] = 1
                elif feat_type == 'Aromatic':
                    flags["aromatic"][atom_idx] = 1
                elif feat_type == 'Hydrophobe':
                    flags["hydrophobe"][atom_idx] = 1
                    
    except Exception as e:
        # If pharmacophore analysis fails (e.g., for large protein structures),
        # return default values (all zeros) and continue
        print(f"Warning: Pharmacophore analysis failed for molecule with {num_atoms} atoms: {e}")
        print("Using default pharmacophore flags (all zeros)")
        # flags already initialized to zeros, so we can just return them
    
    return flags


# Set the number of features as an attribute for the lifter registry
bond_cross_lift.num_features = NUM_FEATURES
