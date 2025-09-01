import torch
from torch_geometric.data import Data

from etnn.combinatorial_data import Cell

# Number of features for the supercell (binding-focused molecular complex features)
NUM_FEATURES = 12


def supercell_lift(graph: Data, **kwargs) -> set[Cell]:
    """
    Return the entire protein-ligand complex as a single supercell with binding-focused features.
    
    This BindingNet-specific supercell lifter creates a single top-dimensional cell containing
    all nodes in the protein-ligand complex, with features that capture binding interface
    properties crucial for affinity prediction in pre-docked complexes.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.
        Must contain:
        - x: node features
        - origin_nodes: tensor indicating ligand (0) vs protein (1) nodes
        - pos: 3D coordinates (required for binding analysis)

    Returns
    -------
    set[Cell]
        A singleton set containing a frozenset of all node indices and a feature
        vector with binding interface properties.

    Features
    --------
    The supercell features include:
    1. Number of interface contacts (< 4.0 Å)
    2. Average contact distance at interface
    3. Minimum ligand-protein distance
    4. Contact density (contacts per ligand atom)
    5. Ligand volume span
    6. Pocket volume span  
    7. Volume complementarity ratio
    8. Interface compactness
    9. Ligand centroid distance to pocket
    10. Interface surface area estimate
    11. Number of ligand atoms
    12. Number of protein atoms
    """

    if (not hasattr(graph, "x")) or (graph.x is None):
        raise ValueError("The given graph does not have a feature matrix 'x'!")
    
    if (not hasattr(graph, "pos")) or (graph.pos is None):
        raise ValueError("The given graph does not have 3D coordinates 'pos' required for binding analysis!")
        
    if (not hasattr(graph, "origin_nodes")) or (graph.origin_nodes is None):
        raise ValueError("The given graph does not have 'origin_nodes' to distinguish ligand/protein atoms!")
    
    num_nodes = graph.x.size(0)
    if num_nodes < 2:
        return set()
    
    # Extract ligand and protein masks
    origin_nodes = graph.origin_nodes
    ligand_mask = (origin_nodes == 0)
    protein_mask = (origin_nodes == 1)
    
    num_ligand = int(ligand_mask.sum().item())
    num_protein = int(protein_mask.sum().item())
    
    if num_ligand == 0 or num_protein == 0:
        # Fallback features for edge cases
        return {(frozenset(range(num_nodes)), (0.0,) * NUM_FEATURES)}  # type: ignore
    
    # Extract positions
    pos = graph.pos
    ligand_pos = pos[ligand_mask]  # Shape: (num_ligand, 3)
    protein_pos = pos[protein_mask]  # Shape: (num_protein, 3)
    
    # ================================================================
    # BINDING INTERFACE ANALYSIS
    # ================================================================
    
    # 1. Interface contact analysis
    lig_prot_distances = torch.cdist(ligand_pos, protein_pos, p=2)  # (num_ligand, num_protein)
    contact_threshold = 4.0  # Angstroms
    contact_mask = (lig_prot_distances < contact_threshold)
    
    num_contacts = contact_mask.sum().float()
    if num_contacts > 0:
        avg_contact_distance = lig_prot_distances[contact_mask].mean()
    else:
        avg_contact_distance = torch.tensor(contact_threshold)  # No contacts
    
    # 2. Minimum distances and proximity
    min_lig_prot_distance = lig_prot_distances.min()
    contact_density = num_contacts / num_ligand  # Contacts per ligand atom
    
    # ================================================================
    # GEOMETRIC COMPLEMENTARITY
    # ================================================================
    
    # 3. Volume estimates using bounding box approach
    ligand_min = ligand_pos.min(dim=0)[0]
    ligand_max = ligand_pos.max(dim=0)[0]
    ligand_spans = ligand_max - ligand_min
    ligand_volume_span = ligand_spans.prod()
    
    protein_min = protein_pos.min(dim=0)[0]
    protein_max = protein_pos.max(dim=0)[0]
    protein_spans = protein_max - protein_min
    protein_volume_span = protein_spans.prod()
    
    # Volume complementarity ratio
    volume_complementarity = ligand_volume_span / (protein_volume_span + 1e-6)
    
    # ================================================================
    # INTERFACE GEOMETRY AND ORGANIZATION
    # ================================================================
    
    # 4. Interface compactness (focus on binding region)
    # Get atoms involved in contacts
    ligand_contact_indices = contact_mask.any(dim=1)  # Ligand atoms with contacts
    protein_contact_indices = contact_mask.any(dim=0)  # Protein atoms with contacts
    
    if ligand_contact_indices.any() and protein_contact_indices.any():
        interface_ligand_pos = ligand_pos[ligand_contact_indices]
        interface_protein_pos = protein_pos[protein_contact_indices]
        interface_atoms = torch.cat([interface_ligand_pos, interface_protein_pos], dim=0)
        
        # Compute interface compactness
        interface_centroid = interface_atoms.mean(dim=0)
        radial_distances = torch.norm(interface_atoms - interface_centroid, dim=1)
        interface_compactness = 1.0 / (1.0 + radial_distances.std().item())
        
        # Interface surface area estimate (convex hull approximation)
        interface_surface_area = _estimate_interface_surface_area(interface_ligand_pos, interface_protein_pos)
    else:
        interface_compactness = 0.0
        interface_surface_area = 0.0
    
    # 5. Centroid-to-centroid distance
    ligand_centroid = ligand_pos.mean(dim=0)
    protein_centroid = protein_pos.mean(dim=0)
    centroid_distance = torch.norm(ligand_centroid - protein_centroid)
    
    # ================================================================
    # ASSEMBLE FEATURE VECTOR WITH NORMALIZATION
    # ================================================================
    
    # Normalize features to prevent massive scaling issues
    # Apply log normalization to volume-based features to reduce their scale
    import math
    
    raw_features = [
        num_contacts.item(),                    # 1. Number of interface contacts
        avg_contact_distance.item(),            # 2. Average contact distance
        min_lig_prot_distance.item(),          # 3. Minimum ligand-protein distance
        contact_density.item(),                 # 4. Contact density (contacts/ligand_atom)
        ligand_volume_span.item(),             # 5. Ligand volume span
        protein_volume_span.item(),            # 6. Protein pocket volume span
        volume_complementarity.item(),          # 7. Volume complementarity ratio
        interface_compactness,                  # 8. Interface compactness
        centroid_distance.item(),              # 9. Ligand-protein centroid distance
        interface_surface_area,                 # 10. Interface surface area estimate
        float(num_ligand),                     # 11. Number of ligand atoms
        float(num_protein),                    # 12. Number of protein atoms
    ]
    
    # Apply feature-specific normalization to prevent scaling issues
    normalized_features = [
        raw_features[0] / 100.0,                           # 1. Scale contacts to ~0-10 range
        raw_features[1],                                   # 2. Distance already reasonable scale
        raw_features[2],                                   # 3. Distance already reasonable scale  
        raw_features[3],                                   # 4. Ratio already reasonable scale
        math.log(raw_features[4] + 1.0),                  # 5. Log-scale ligand volume
        math.log(raw_features[5] + 1.0),                  # 6. Log-scale protein volume
        raw_features[6],                                   # 7. Ratio already reasonable scale
        raw_features[7],                                   # 8. Compactness already reasonable scale
        raw_features[8],                                   # 9. Distance already reasonable scale
        math.log(raw_features[9] + 1.0),                  # 10. Log-scale surface area
        raw_features[10] / 100.0,                         # 11. Scale ligand atoms to ~0-10 range
        raw_features[11] / 1000.0,                        # 12. Scale protein atoms to ~0-10 range
    ]
    
    features = tuple(normalized_features)
    
    # Create supercell containing all nodes
    all_nodes = frozenset(range(num_nodes))
    
    return {(all_nodes, features)}  # type: ignore


def _estimate_interface_surface_area(ligand_pos: torch.Tensor, protein_pos: torch.Tensor) -> float:
    """
    Estimate the interface surface area between ligand and protein using a simplified approach.
    
    This function approximates the buried surface area by computing the overlap region
    between ligand and protein volumes using their convex hull projections.
    
    Parameters
    ----------
    ligand_pos : torch.Tensor
        3D coordinates of ligand atoms at the interface. Shape: (n_ligand_interface, 3)
    protein_pos : torch.Tensor  
        3D coordinates of protein atoms at the interface. Shape: (n_protein_interface, 3)
        
    Returns
    -------
    float
        Estimated interface surface area (simplified metric)
    """
    if ligand_pos.shape[0] == 0 or protein_pos.shape[0] == 0:
        return 0.0
    
    # Simple approach: compute overlapping volume using bounding box intersection
    # This is a crude approximation but computationally efficient
    
    # Get bounding boxes
    lig_min = ligand_pos.min(dim=0)[0]
    lig_max = ligand_pos.max(dim=0)[0]
    prot_min = protein_pos.min(dim=0)[0]
    prot_max = protein_pos.max(dim=0)[0]
    
    # Compute intersection of bounding boxes
    intersect_min = torch.max(lig_min, prot_min)
    intersect_max = torch.min(lig_max, prot_max)
    
    # Check if there's an intersection
    intersection_dims = intersect_max - intersect_min
    if (intersection_dims > 0).all():
        # Surface area approximation: use the largest two dimensions of intersection
        sorted_dims, _ = torch.sort(intersection_dims, descending=True)
        surface_area = sorted_dims[0] * sorted_dims[1]  # Largest two dimensions
        return surface_area.item()
    else:
        return 0.0


def get_supercell_num_features(graph: Data) -> int:
    """Determine number of features for BindingNet supercell"""
    return NUM_FEATURES


# Set the num_features attribute for compatibility
supercell_lift.num_features = NUM_FEATURES  # type: ignore
