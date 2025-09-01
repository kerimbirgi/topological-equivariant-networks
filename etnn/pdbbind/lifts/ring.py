"""
Ring lifter for PDBBind dataset - extracts rings from ligand molecules only.
Based on the QM9 ring lifter but adapted for ligand-protein systems.
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from etnn.combinatorial_data import Cell

# Number of features per ring
NUM_FEATURES = 4  # ring_size, is_aromatic, has_heteroatom, is_saturated


def ring_lift(graph: Data, **kwargs) -> set[Cell]:
    """
    Identify rings in ligand molecules only.
    
    This function identifies rings in the ligand part of a protein-ligand complex
    and returns them as a set of cells. Each cell represents a ring and consists 
    of a frozenset of node indices and a feature vector. Only processes ligand 
    atoms since proteins typically don't have rings that are relevant for binding.
    
    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.
        Must have 'origin_nodes' attribute to distinguish ligand (0) from protein (1).
        Must have 'ligand_mol' attribute containing the RDKit molecule object.
    **kwargs
        Additional keyword arguments (unused but maintained for compatibility).
    
    Returns
    -------
    set[Cell]
        A set of cells, each representing a ring in the ligand. Each cell consists 
        of a frozenset of node indices and a feature vector.
    
    Raises
    ------
    ValueError
        If the input graph does not contain required attributes.
    
    Notes
    -----
    - Only processes ligand atoms (origin_nodes == 0)
    - Requires RDKit molecule object in graph.ligand_mol
    - Returns minimal rings (no rings that contain smaller rings)
    - Ring features: size, aromaticity, heteroatom presence, saturation
    """
    cells = set()
    
    # Check for required attributes
    if not hasattr(graph, 'origin_nodes') or graph.origin_nodes is None:
        print("Warning: Graph missing origin_nodes, cannot identify ligand atoms")
        return cells
    
    if not hasattr(graph, 'ligand_mol') or graph.ligand_mol is None:
        print("Warning: Graph missing ligand_mol, cannot extract ring information")
        return cells
    
    # Get ligand molecule and atom indices
    ligand_mol = graph.ligand_mol
    origin_nodes = graph.origin_nodes
    ligand_mask = (origin_nodes == 0)
    ligand_atom_indices = torch.nonzero(ligand_mask).flatten().tolist()
    
    if len(ligand_atom_indices) == 0:
        print("Warning: No ligand atoms found")
        return cells
    
    try:
        # Get ring information from RDKit
        ring_info = ligand_mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        # Process each ring
        for ring in atom_rings:
            # Convert local molecule indices to global graph indices
            # Note: ring indices are in molecule coordinate system
            global_ring_indices = []
            for mol_idx in ring:
                if mol_idx < len(ligand_atom_indices):
                    global_ring_indices.append(ligand_atom_indices[mol_idx])
            
            # Only process rings with at least 3 atoms
            if len(global_ring_indices) >= 3:
                # Compute ring features using molecule indices
                feature_vector = compute_ring_features(ring, ligand_mol)
                # Create cell with global indices
                cells.add((frozenset(global_ring_indices), feature_vector))
        
        # Filter out rings that contain simpler rings within themselves
        # (keep only minimal rings)
        filtered_cells = {
            cell
            for cell in cells
            if not any(cell[0] > other_cell[0] for other_cell in cells if cell != other_cell)
        }
        
        return filtered_cells
        
    except Exception as e:
        print(f"Warning: Failed to extract ligand rings: {e}")
        return cells


def compute_ring_features(ring_indices: tuple, molecule: Chem.Mol) -> tuple[float]:
    """
    Compute features for a ring in a ligand molecule.
    
    This function computes features for a ring in a molecule. The features include 
    the ring size, whether the ring is aromatic, whether the ring has a heteroatom, 
    and whether the ring is saturated.
    
    Parameters
    ----------
    ring_indices : tuple[int]
        A tuple of atom indices representing the ring (in molecule coordinates).
    molecule : Chem.Mol
        The RDKit molecule object representing the ligand.
    
    Returns
    -------
    tuple[float]
        A tuple of features for the ring:
        - ring_size: Number of atoms in the ring
        - is_aromatic: 1.0 if all atoms in ring are aromatic, 0.0 otherwise  
        - has_heteroatom: 1.0 if ring contains non-C/H atoms, 0.0 otherwise
        - is_saturated: 1.0 if all atoms are SP3 hybridized, 0.0 otherwise
    
    Notes
    -----
    The function uses the RDKit library to extract ring information from the molecule.
    """
    try:
        # Get atom objects for the ring
        ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring_indices]
        
        # Feature 1: Ring size
        ring_size = float(len(ring_indices))
        
        # Feature 2: Aromaticity (all atoms in ring must be aromatic)
        is_aromatic = float(all(atom.GetIsAromatic() for atom in ring_atoms))
        
        # Feature 3: Heteroatom presence (any non-carbon, non-hydrogen atom)
        has_heteroatom = float(any(atom.GetSymbol() not in ("C", "H") for atom in ring_atoms))
        
        # Feature 4: Saturation (all atoms SP3 hybridized)
        is_saturated = float(
            all(atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in ring_atoms)
        )
        
        return (ring_size, is_aromatic, has_heteroatom, is_saturated)
        
    except Exception as e:
        print(f"Warning: Error computing ring features for ring {ring_indices}: {e}")
        # Return default features on error
        return (float(len(ring_indices)), 0.0, 0.0, 0.0)


# Set the number of features as an attribute for the lifter registry
ring_lift.num_features = NUM_FEATURES
