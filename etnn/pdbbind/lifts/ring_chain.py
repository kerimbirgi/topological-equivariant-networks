"""
Heterogeneous lifter for rank 2: Ligand Rings + Protein Chains
"""
import torch
from torch_geometric.data import Data
from etnn.combinatorial_data import Cell
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Feature dimensions must match for both rings and chains
NUM_FEATURES = 10  # Adjust based on your feature design

def ring_chain_lift(graph: Data, **kwargs) -> set[Cell]:
    """
    Create rank 2 cells containing both ligand rings and protein chains.
    
    Returns
    -------
    set[Cell]
        Set of cells where each cell is either a ligand ring or protein chain
    """
    cells = set()
    
    # Check required attributes
    if not hasattr(graph, 'origin_nodes') or graph.origin_nodes is None:
        print("Warning: Graph missing origin_nodes, cannot distinguish ligand/protein")
        return cells
    
    # Extract ligand and protein molecules if available
    ligand_mol = getattr(graph, 'ligand_mol', None)
    protein_mol = getattr(graph, 'protein_mol', None)
    
    # Get ligand and protein atom indices
    origin_nodes = graph.origin_nodes
    ligand_mask = (origin_nodes == 0)
    protein_mask = (origin_nodes == 1)
    
    ligand_atoms = torch.nonzero(ligand_mask).flatten().tolist()
    protein_atoms = torch.nonzero(protein_mask).flatten().tolist()
    
    # Extract ligand rings
    if ligand_mol is not None:
        ligand_rings = extract_ligand_rings(ligand_mol, ligand_atoms, graph)
        cells.update(ligand_rings)
    
    # Extract protein chains/secondary structures  
    protein_chains = extract_protein_chains(protein_atoms, graph)
    cells.update(protein_chains)
    
    return cells

def extract_ligand_rings(mol: Chem.Mol, ligand_atom_indices: list, graph: Data) -> set[Cell]:
    """Extract ring systems from ligand molecule."""
    cells = set()
    
    try:
        # Get ring information from RDKit
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        for ring in atom_rings:
            # Convert local molecule indices to global graph indices
            global_ring_indices = [ligand_atom_indices[i] for i in ring if i < len(ligand_atom_indices)]
            
            if len(global_ring_indices) >= 3:  # Valid ring
                ring_features = compute_ring_features(ring, mol, graph, global_ring_indices)
                cells.add((frozenset(global_ring_indices), tuple(ring_features)))
    
    except Exception as e:
        print(f"Warning: Failed to extract ligand rings: {e}")
    
    return cells

def extract_protein_chains(protein_atom_indices: list, graph: Data) -> set[Cell]:
    """Extract protein chain/secondary structure segments."""
    cells = set()
    
    try:
        # Simple chain segmentation based on connectivity
        # In a real implementation, you'd use protein structure info
        
        if len(protein_atom_indices) < 5:
            return cells
        
        # Create chains of connected protein atoms (simplified)
        # This is a placeholder - real implementation would use:
        # - Secondary structure (alpha helices, beta sheets)
        # - Sequence information  
        # - Spatial clustering
        
        chain_size = min(20, len(protein_atom_indices) // 4)  # Adaptive chain size
        
        for i in range(0, len(protein_atom_indices), chain_size):
            chain_atoms = protein_atom_indices[i:i+chain_size]
            if len(chain_atoms) >= 5:  # Minimum chain size
                chain_features = compute_chain_features(chain_atoms, graph)
                cells.add((frozenset(chain_atoms), tuple(chain_features)))
    
    except Exception as e:
        print(f"Warning: Failed to extract protein chains: {e}")
    
    return cells

def compute_ring_features(ring_indices: tuple, mol: Chem.Mol, graph: Data, global_indices: list) -> list:
    """Compute features for a ligand ring."""
    features = [0.0] * NUM_FEATURES
    
    try:
        # Feature 1: Ring size
        features[0] = len(ring_indices)
        
        # Feature 2: Aromaticity
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring_indices if i < mol.GetNumAtoms()]
        features[1] = float(all(atom.GetIsAromatic() for atom in ring_atoms))
        
        # Feature 3: Average ring atom electronegativity
        electronegativities = []
        for atom in ring_atoms:
            atomic_num = atom.GetAtomicNum()
            # Simplified electronegativity lookup
            en_map = {1: 2.2, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16}
            electronegativities.append(en_map.get(atomic_num, 2.0))
        features[2] = sum(electronegativities) / len(electronegativities) if electronegativities else 2.0
        
        # Feature 4: Ring planarity (simplified)
        if len(global_indices) >= 3:
            positions = graph.pos[global_indices]
            # Compute variance in z-direction as planarity measure
            features[3] = float(positions[:, 2].var())
        
        # Features 5-10: Additional ring properties
        features[4] = float(len([a for a in ring_atoms if a.GetAtomicNum() == 7]))  # N count
        features[5] = float(len([a for a in ring_atoms if a.GetAtomicNum() == 8]))  # O count
        features[6] = float(len([a for a in ring_atoms if a.GetAtomicNum() == 16])) # S count
        features[7] = 1.0  # Ring type marker (1 = ligand ring)
        features[8] = 0.0  # Reserved
        features[9] = 0.0  # Reserved
        
    except Exception as e:
        print(f"Warning: Error computing ring features: {e}")
        features = [0.0] * NUM_FEATURES
    
    return features

def compute_chain_features(chain_atoms: list, graph: Data) -> list:
    """Compute features for a protein chain segment."""
    features = [0.0] * NUM_FEATURES
    
    try:
        # Feature 1: Chain length
        features[0] = len(chain_atoms)
        
        # Feature 2: Always 0 for protein chains (vs 1 for ligand rings)
        features[1] = 0.0
        
        # Feature 3: Average atom electronegativity in chain
        if hasattr(graph, 'x') and graph.x is not None:
            atom_features = graph.x[chain_atoms]
            # Assuming first feature is atomic number
            atomic_nums = atom_features[:, 0] if atom_features.shape[1] > 0 else [6]
            en_map = {1: 2.2, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16}
            electronegativities = [en_map.get(int(num), 2.0) for num in atomic_nums]
            features[2] = sum(electronegativities) / len(electronegativities)
        
        # Feature 4: Chain spatial extent
        if len(chain_atoms) >= 2:
            positions = graph.pos[chain_atoms]
            extent = (positions.max(dim=0)[0] - positions.min(dim=0)[0]).norm()
            features[3] = float(extent)
        
        # Features 5-10: Additional chain properties
        features[4] = 0.0  # N count (placeholder)
        features[5] = 0.0  # O count (placeholder)  
        features[6] = 0.0  # S count (placeholder)
        features[7] = 0.0  # Chain type marker (0 = protein chain)
        features[8] = 0.0  # Reserved
        features[9] = 0.0  # Reserved
        
    except Exception as e:
        print(f"Warning: Error computing chain features: {e}")
        features = [0.0] * NUM_FEATURES
    
    return features

# Set feature count for compatibility
ring_chain_lift.num_features = NUM_FEATURES
