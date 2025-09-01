import os
import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')   # hide warnings, keep errors
import argparse
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import sys
import numpy as np
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import math

logger = logging.getLogger(__name__)

def cosine_envelope(d, r_cut):
    # d: (E,1) or (E,)
    x = d / r_cut
    env = 0.5 * (torch.cos(math.pi * x).clamp(min=-1.0, max=1.0) + 1.0)
    env = env * (d <= r_cut)
    return env

def rbf_expand(d, r_cut, M=16, eps=1e-8):
    """
    d: (E,1) or (E,) distances in Å
    returns: (E, M) Gaussian RBFs with cosine envelope.
    """
    if d.ndim == 1: d = d.unsqueeze(1)     # (E,1)
    centers = torch.linspace(0.0, r_cut, M, device=d.device).view(1, M)  # (1,M)
    delta = r_cut / max(M-1, 1)
    gamma = 1.0 / (delta**2 + eps)
    rbf = torch.exp(-gamma * (d - centers)**2)  # (E,M)
    rbf = rbf * cosine_envelope(d, r_cut)       # smooth cutoff
    return rbf

_FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FEAT_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF)

def _atom_flags_pharmacophore(mol: Chem.Mol) -> dict[str, torch.Tensor]:
    """
    Returns boolean (uint8) tensors of shape [N] for:
      donor, acceptor, aromatic, cationic, anionic, hydrophobe
    Notes:
      - 'aromatic' uses RDKit atom aromaticity.
      - 'cationic'/'anionic' are from formal charge (fallback, simple but robust).
      - 'hydrophobe' is a very coarse heuristic: non-polar atom types (C,S,F,Cl,Br,I and not charged).
    """
    N = mol.GetNumAtoms()
    donors = np.zeros(N, dtype=np.uint8)
    acceptors = np.zeros(N, dtype=np.uint8)

    # RDKit pharmacophore features (donor/acceptor)
    for f in _FEAT_FACTORY.GetFeaturesForMol(mol):
        fam = f.GetFamily()
        if fam == "Donor":
            for idx in f.GetAtomIds():
                donors[idx] = 1
        elif fam == "Acceptor":
            for idx in f.GetAtomIds():
                acceptors[idx] = 1

    aromatic = np.array([int(a.GetIsAromatic()) for a in mol.GetAtoms()], dtype=np.uint8)
    charges  = np.array([int(a.GetFormalCharge() or 0) for a in mol.GetAtoms()], dtype=np.int8)

    cationic = (charges > 0).astype(np.uint8)
    anionic  = (charges < 0).astype(np.uint8)

    # very coarse hydrophobe tag: carbon/non-polar halogens/sulfur w/ no formal charge
    non_polar_atomic_nums = {6, 9, 17, 35, 53, 16}  # C, F, Cl, Br, I, S
    hydrophobe = np.array([
        1 if (a.GetAtomicNum() in non_polar_atomic_nums and charges[i] == 0) else 0
        for i, a in enumerate(mol.GetAtoms())
    ], dtype=np.uint8)

    return {
        "donor":      torch.from_numpy(donors),
        "acceptor":   torch.from_numpy(acceptors),
        "aromatic":   torch.from_numpy(aromatic),
        "cationic":   torch.from_numpy(cationic),
        "anionic":    torch.from_numpy(anionic),
        "hydrophobe": torch.from_numpy(hydrophobe),
    }

HYB_TYPES = ['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2']
# Bond type to index mapping (used for one-hot encoding)
# Covers the most common bond types encountered in molecular data
BOND_TYPES = {
    BT.SINGLE: 0,      # Standard single bond
    BT.DOUBLE: 1,      # Standard double bond  
    BT.TRIPLE: 2,      # Standard triple bond
    BT.AROMATIC: 3,    # Aromatic bond
    BT.ONEANDAHALF: 4, # 1.5 bond (common in resonance structures)
    BT.TWOANDAHALF: 5, # 2.5 bond
    BT.UNSPECIFIED: 6  # Fallback for unknown/unspecified bonds
}

# Fallback bond type for any bonds not in BOND_TYPES
FALLBACK_BOND_TYPE = BT.UNSPECIFIED


# ==== RBF config ====
M_RBF = 16          # number of radial bases
R_CUT_BOND = 3.0    # covers covalent bond lengths comfortably

# Expected feature dimensions for validation
EXPECTED_NODE_FEATURES = 13         # unchanged
EXPECTED_EDGE_FEATURES = 9 + M_RBF

PREFER_MOL2 = True
def process_ligand_sdf(sdf_path: str) -> Data:
    """
    Process ligand SDF files to create PyTorch Geometric Data objects.
    
    Node Features (13D):
    - Atomic number (1D): Element type
    - Degree (1D): Number of bonded neighbors  
    - Formal charge (1D): Molecular charge state
    - Atomic mass (1D): Element mass
    - In ring flag (1D): Binary indicator for ring membership
    - Aromatic flag (1D): Binary indicator for aromaticity
    - Hybridization one-hot (7D): SP, SP2, SP3, SP3D, SP3D2, S, UNSPECIFIED
    
    Edge Features (25D = 9 + 16):
    - Bond type one-hot (7D): SINGLE, DOUBLE, TRIPLE, AROMATIC, ONEANDAHALF, TWOANDAHALF, UNSPECIFIED
    - Conjugation flag (1D): Binary indicator for conjugated bonds
    - Ring bond flag (1D): Binary indicator for bonds in rings
    - Distance RBF (16D): Radial basis functions encoding bond length (cutoff=3.0Å)
    
    Returns:
        Data object with x (node features), pos (3D coordinates), edge_index, edge_attr
    """
    if not os.path.exists(sdf_path):
        print(f"Path doesnt exist {sdf_path}")
        #exit()
        return None

    if PREFER_MOL2:
        try:
            mol2_path = sdf_path.replace(".sdf", ".mol2")
            mol = Chem.MolFromMol2File(mol2_path, sanitize=True, removeHs=False)
        except Exception as e:
            try:
                supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
                if len(supplier) > 1:
                    raise IndexError(f"Unexpexcted number of Molecules in ligand.sdf: {sdf_path}")
                mol = supplier[0] # We process single molecule ligands, no disconnected ones.
                if mol is None:
                    raise ValueError(f"Could not parse ligand SDF file: {sdf_path}")
            except Exception as e:
                raise ValueError(f"Could not parse ligand sdf or mol2 file: {sdf_path}")
    else:
        try:
            supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
            if len(supplier) > 1:
                raise IndexError(f"Unexpexcted number of Molecules in ligand.sdf: {sdf_path}")
            mol = supplier[0] # We process single molecule ligands, no disconnected ones.
            if mol is None:
                raise ValueError(f"Could not parse ligand SDF file: {sdf_path}")
        except Exception as e:
            try:
                mol2_path = sdf_path.replace(".sdf", ".mol2")
                mol = Chem.MolFromMol2File(mol2_path, sanitize=True, removeHs=False)
                if mol is None:
                    raise ValueError(f"Could not parse ligand mol2 file: {sdf_path}")
            except Exception as e:
                raise ValueError(f"Could not parse ligand sdf or mol2 file: {sdf_path}")



    # Node features
    
    # Precompute these thingsonce per molecule
    #donors, acceptors = donor_acceptor_masks(mol)
    #logp_contribs     = Crippen.MolAtomLogP(mol)
    #radii = rdFreeSASA.ClassicAtomicRadii     # dictionary of VDWs
    #rdFreeSASA.CalcSASA(mol, radii)

    atom_features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        num  = atom.GetAtomicNum()       # Atomic number
        deg  = atom.GetDegree()          # Neighbor degree
        chg  = atom.GetFormalCharge()    # Formal charge
        mass = atom.GetMass()            # Mass
        in_ring = 1 if atom.IsInRing() else 0 # In ring flag
        aro = 1 if atom.GetIsAromatic() else 0 # Aromaticity flag
        
        # Hydrogen donor/acceptor flags
        #is_donor     = 1 if idx in donors     else 0
        #is_acceptor  = 1 if idx in acceptors  else 0
        #hydrophob     = 1 if logp_contribs[idx] > 0 else 0 # hydrophobicity
        #sasa = float(mol.GetAtomWithIdx(idx).GetProp('SASA'))   # Å², solvent accessible surface area


        # Hybridization one-hot
        hyb  = atom.GetHybridization().name 
        hyb_idx = HYB_TYPES.index(hyb)
        hyb_oh  = [1 if i == hyb_idx else 0 for i in range(len(HYB_TYPES))]

        atom_features.append([num, deg, chg, mass, in_ring, aro] + hyb_oh)

    x = torch.tensor(atom_features, dtype=torch.float)  # [N, F_node]

    # 3D positions
    conf = mol.GetConformer()
    pos = torch.tensor([
        [conf.GetAtomPosition(i).x,
        conf.GetAtomPosition(i).y,
        conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ], dtype=torch.float)                            # [N, 3]

    # Edge indices and attributes (directed edges + one-hot bond type)
    rows, cols, edge_feat_list = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # add both directions to make the graph effectively undirected
        rows += [start, end]
        cols += [end, start]

        # Bond-type one-hot (safe lookup with fallback to UNSPECIFIED)
        bt_idx = BOND_TYPES.get(bond.GetBondType(), BOND_TYPES[FALLBACK_BOND_TYPE])
        bt_oh = [int(i == bt_idx) for i in range(len(BOND_TYPES))]
        #edge_types += 2 * [BOND_TYPES[bond.GetBondType()]]

        # conjugation flag
        is_conj = [1 if bond.GetIsConjugated() else 0]

        # ring edge flag
        in_ring = [int(bond.IsInRing())]

        p1, p2 = mol.GetConformer().GetAtomPosition(start), mol.GetConformer().GetAtomPosition(end)
        length_val = p1.Distance(p2)
        length_t = torch.tensor([length_val], dtype=torch.float32)
        rbf_len   = rbf_expand(length_t, r_cut=R_CUT_BOND, M=M_RBF)

        feats_base = torch.tensor(bt_oh + is_conj + in_ring, dtype=torch.float32).unsqueeze(0)  # (1,9)
        feats = torch.cat([feats_base, rbf_len], dim=1).squeeze(0)

        edge_feat_list += [feats.tolist(), feats.tolist()]
        

    edge_index = torch.tensor([rows, cols], dtype=torch.long)  # [2, E]
    
    # Handle case where molecule has no bonds
    if len(edge_feat_list) == 0:
        # Create empty tensor with correct feature dimension
        edge_attr = torch.empty(0, EXPECTED_EDGE_FEATURES, dtype=torch.float32)
    else:
        edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)

    perm = (edge_index[0] * mol.GetNumAtoms() + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr  = edge_attr[perm]

    # Validation: Check expected dimensions
    if x.shape[1] != EXPECTED_NODE_FEATURES:
        logger.warning(f"Ligand {sdf_path}: Expected {EXPECTED_NODE_FEATURES} node features, got {x.shape[1]}")
        raise ValueError(f"Ligand node feature dimension mismatch: expected {EXPECTED_NODE_FEATURES}, got {x.shape[1]}")
    
    if edge_attr.shape[0] > 0 and edge_attr.shape[1] != EXPECTED_EDGE_FEATURES:
        logger.warning(f"Ligand {sdf_path}: Expected {EXPECTED_EDGE_FEATURES} edge features, got {edge_attr.shape[1]}")
        raise ValueError(f"Ligand edge feature dimension mismatch: expected {EXPECTED_EDGE_FEATURES}, got {edge_attr.shape[1]}")
    
    if x.shape[0] == 0:
        logger.warning(f"Ligand {sdf_path}: No atoms in molecule")
        raise ValueError("Ligand has no atoms")
    
    if pos.shape != (x.shape[0], 3):
        logger.warning(f"Ligand {sdf_path}: Position shape mismatch, expected ({x.shape[0]}, 3), got {pos.shape}")
        raise ValueError(f"Ligand position shape mismatch: expected ({x.shape[0]}, 3), got {pos.shape}")

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mol=mol
    )

def process_protein_pdb_ligand_style(pdb_path: str) -> Data:
    """
    Process protein PDB files to create PyTorch Geometric Data objects with identical 
    feature format to process_ligand_sdf for seamless merging.
    
    Node Features (13D):
    - Atomic number (1D): Element type
    - Degree (1D): Number of bonded neighbors  
    - Formal charge (1D): Molecular charge state
    - Atomic mass (1D): Element mass
    - In ring flag (1D): Binary indicator for ring membership
    - Aromatic flag (1D): Binary indicator for aromaticity
    - Hybridization one-hot (7D): SP, SP2, SP3, SP3D, SP3D2, S, UNSPECIFIED
    
    Edge Features (25D = 9 + 16):
    - Bond type one-hot (7D): SINGLE, DOUBLE, TRIPLE, AROMATIC, ONEANDAHALF, TWOANDAHALF, UNSPECIFIED
    - Conjugation flag (1D): Binary indicator for conjugated bonds
    - Ring bond flag (1D): Binary indicator for bonds in rings
    - Distance RBF (16D): Radial basis functions encoding bond length (cutoff=3.0Å)
    
    Returns:
        Data object with x (node features), pos (3D coordinates), edge_index, edge_attr
        Compatible with ligand graphs for merging operations.
    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit failed to parse PDB: {pdb_path}")

    # Sanitize the molecule, if possible, otherwise continue
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    # Node features ---------------------------------------------------------
    atom_features = []
    for atom in mol.GetAtoms():
        num  = atom.GetAtomicNum()
        deg  = atom.GetDegree()
        chg  = atom.GetFormalCharge() or 0
        mass = atom.GetMass()
        in_ring = int(atom.IsInRing())
        aro = int(atom.GetIsAromatic())
        hyb_name = atom.GetHybridization().name
        hyb_idx  = HYB_TYPES.index(hyb_name) if hyb_name in HYB_TYPES else 0
        hyb_oh   = [int(i == hyb_idx) for i in range(len(HYB_TYPES))]
        atom_features.append([num, deg, chg, mass, in_ring, aro] + hyb_oh)
    x = torch.tensor(atom_features, dtype=torch.float32)

    # Edge features ---------------------------------------------------------
    rows, cols, edge_feat_list = [], [], []
    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        bt_idx = BOND_TYPES.get(bond.GetBondType(), BOND_TYPES[FALLBACK_BOND_TYPE])
        bt_oh = [int(i == bt_idx) for i in range(len(BOND_TYPES))]
        is_conj = [int(bond.GetIsConjugated())]
        in_ring = [int(bond.IsInRing())]
        p1, p2 = conf.GetAtomPosition(start), conf.GetAtomPosition(end)
        # scalar length -> RBF
        length_val = p1.Distance(p2)
        length_t = torch.tensor([length_val], dtype=torch.float32)        # (1,)
        rbf_len   = rbf_expand(length_t, r_cut=R_CUT_BOND, M=M_RBF)       # (1, M_RBF)

        # categorical part: 7 bond types + 1 conjugation + 1 ring = 9
        feats_base = torch.tensor(bt_oh + is_conj + in_ring, dtype=torch.float32).unsqueeze(0)  # (1, 9)

        # final per-direction edge feature: (9 + M_RBF,)
        feats = torch.cat([feats_base, rbf_len], dim=1).squeeze(0)

        edge_feat_list += [feats.tolist(), feats.tolist()]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    # Handle case where molecule has no bonds
    if len(edge_feat_list) == 0:
        # Create empty tensor with correct feature dimension
        edge_attr = torch.empty(0, EXPECTED_EDGE_FEATURES, dtype=torch.float32)
    else:
        edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)

    perm = (edge_index[0] * mol.GetNumAtoms() + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr  = edge_attr[perm]

    pos = torch.tensor([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=torch.float32)

    # Validation: Check expected dimensions
    if x.shape[1] != EXPECTED_NODE_FEATURES:
        logger.warning(f"Protein {pdb_path}: Expected {EXPECTED_NODE_FEATURES} node features, got {x.shape[1]}")
        raise ValueError(f"Protein node feature dimension mismatch: expected {EXPECTED_NODE_FEATURES}, got {x.shape[1]}")
    
    if edge_attr.shape[0] > 0 and edge_attr.shape[1] != EXPECTED_EDGE_FEATURES:
        logger.warning(f"Protein {pdb_path}: Expected {EXPECTED_EDGE_FEATURES} edge features, got {edge_attr.shape[1]}")
        raise ValueError(f"Protein edge feature dimension mismatch: expected {EXPECTED_EDGE_FEATURES}, got {edge_attr.shape[1]}")
    
    if x.shape[0] == 0:
        logger.warning(f"Protein {pdb_path}: No atoms in molecule")
        raise ValueError("Protein has no atoms")
    
    if pos.shape != (x.shape[0], 3):
        logger.warning(f"Protein {pdb_path}: Position shape mismatch, expected ({x.shape[0]}, 3), got {pos.shape}")
        raise ValueError(f"Protein position shape mismatch: expected ({x.shape[0]}, 3), got {pos.shape}")

    return Data(
        x=x, 
        pos=pos,
        edge_index=edge_index, 
        edge_attr=edge_attr,
        mol = mol
    )


def merge_ligand_and_protein(
        ligand:  Data,
        protein: Data,
        connect_cross: bool = False,
        r_cut: float = 5.0,
) -> Data:
    """
    Merge ligand and protein graphs into a single heterogeneous molecular complex.
    
    Merged Node Features (14D = 13 + 1):
    - Original features (13D): Same as individual graphs (atomic_num, degree, charge, mass, in_ring, aromatic, hybridization_oh)
    - Origin flag (1D): 0.0 = ligand atom, 1.0 = protein atom
    
    Merged Edge Features (without cross-connect: 27D = 25 + 2):
    - Original features (25D): Same as individual graphs (bond_type_oh, conjugation, ring_bond, distance_rbf)
    - Edge type one-hot (2D): [1,0] = intra_ligand, [0,1] = intra_protein
    
    Merged Edge Features (with cross-connect: 33D = 25 + 5 + 3):
    - Original features (25D): Same as individual graphs
    - Interaction features (5D): [hbond, salt_bridge, pi_pi, cation_pi, hydrophobe_contact] for cross edges, zeros for intra
    - Edge type one-hot (3D): [1,0,0] = intra_ligand, [0,1,0] = intra_protein, [0,0,1] = inter_molecular
    
    Cross-Edge Interaction Features (when connect_cross=True):
    - H-bond: Donor-acceptor pairs within 3.5Å
    - Salt bridge: Oppositely charged atoms within 4.0Å  
    - π-π stacking: Aromatic-aromatic atoms within 6.0Å
    - Cation-π: Charged-aromatic atoms within 6.0Å
    - Hydrophobic contact: Non-polar atoms within 4.5Å
    
    Parameters:
        connect_cross: If True, adds intermolecular edges within r_cut distance
        r_cut: Distance cutoff for cross edges (default 5.0Å)

    New attributes:
        origin_nodes : (N,)   0 = ligand node,   1 = protein node
        origin_edges : (E,)   0 = ligand edge,   1 = protein edge, 2 = cross edge
    """
    # ------------------------------------------------------------------ #
    # 1. nodes
    # ------------------------------------------------------------------ #
    # Assert required tensors exist (helps type checker and avoids runtime None errors)
    assert ligand.x is not None and ligand.pos is not None and ligand.edge_index is not None and ligand.edge_attr is not None
    assert protein.x is not None and protein.pos is not None and protein.edge_index is not None and protein.edge_attr is not None

    # number of ligand and protein atoms
    num_lig = ligand.x.size(0) 
    num_pro = protein.x.size(0) 

    # Number of interaction features we will append on cross edges when connect_cross=True
    K_INTER = 5  # [hbond, salt_bridge, pi_pi, cation_pi, hydrophobe_contact]



    # concatenate ligand and protein node features and positions
    x   = torch.cat([ligand.x, protein.x], dim=0)          # (N, F)
    pos = torch.cat([ligand.pos, protein.pos], dim=0)      # (N, 3)

    # Atom-level origin flag: 0 = ligand, 1 = protein
    lig_flag = torch.zeros((num_lig, 1), dtype=x.dtype, device=x.device)
    pro_flag = torch.ones((num_pro, 1),  dtype=x.dtype, device=x.device)
    origin_flag = torch.cat([lig_flag, pro_flag], dim=0)   # (N, 1)
    x = torch.cat([x, origin_flag], dim=1)                 # (N, F+1)

    # For tracking the origin of each node, get used as custom attribute in the Data object
    origin_nodes = torch.cat([
        torch.zeros(num_lig, dtype=torch.long),            # ligand nodes
        torch.ones (num_pro, dtype=torch.long)             # protein nodes
    ], dim=0)                                              # (N,) 

    # ------------------------------------------------------------------ #
    # 2. edges  (shift indices of protein by +num_lig)
    # ------------------------------------------------------------------ #
    prot_shift = num_lig
    edge_index_lig = ligand.edge_index                     # already fine
    edge_index_pro = protein.edge_index + prot_shift       # shifted

    # Add edge type features (one-hot encoding)
    # no connect_cross: [intra_ligand, intra_protein]
    # connect_cross:  [intra_ligand, intra_protein, inter_molecular]
    num_lig_edges = ligand.edge_attr.size(0)
    num_pro_edges = protein.edge_attr.size(0)
    
    if connect_cross:
        # [intra_ligand, intra_protein, inter_molecular]
        lig_type = torch.tensor([[1, 0, 0]], dtype=torch.float).repeat(num_lig_edges, 1) # (num_lig_edges, 3)
        pro_type = torch.tensor([[0, 1, 0]], dtype=torch.float).repeat(num_pro_edges, 1) # (num_pro_edges, 3)
        
        # Add interaction feature slots (filled with zeros for intra-molecular edges)
        zeros_inter_lig = torch.zeros((num_lig_edges, K_INTER), dtype=torch.float)
        zeros_inter_pro = torch.zeros((num_pro_edges, K_INTER), dtype=torch.float)
        # intra edges: [original(25)] + [interaction zeros(5)] + [type(3)]
        ligand_edge_attr_enhanced  = torch.cat([ligand.edge_attr,  zeros_inter_lig, lig_type], dim=1)
        protein_edge_attr_enhanced = torch.cat([protein.edge_attr, zeros_inter_pro, pro_type], dim=1)
    else:
        # [intra_ligand, intra_protein]
        lig_type = torch.tensor([[1, 0]], dtype=torch.float).repeat(num_lig_edges, 1) # (num_lig_edges, 2)
        pro_type = torch.tensor([[0, 1]], dtype=torch.float).repeat(num_pro_edges, 1) # (num_pro_edges, 2)
        
        # No interaction features needed when cross connections are disabled
        # intra edges: [original(25)] + [type(2)]
        ligand_edge_attr_enhanced  = torch.cat([ligand.edge_attr, lig_type], dim=1)
        protein_edge_attr_enhanced = torch.cat([protein.edge_attr, pro_type], dim=1)


    
    # Sanity check: dimensions of edge features should be same for protein and ligand
    if ligand_edge_attr_enhanced.size(1) != protein_edge_attr_enhanced.size(1):
        raise ValueError(
            f"Edge feature dimension mismatch after fixes: "
            f"ligand={ligand_edge_attr_enhanced.size(1)} vs protein={protein_edge_attr_enhanced.size(1)}. "
            f"Original ligand edge_attr: {ligand.edge_attr.size(1)}, "
            f"Original protein edge_attr: {protein.edge_attr.size(1)}"
        )

    # Create merged graph edges and attributes, NOTE: Does not include cross edges yet!
    edge_index = torch.cat([edge_index_lig, edge_index_pro], dim=1)
    edge_attr  = torch.cat([ligand_edge_attr_enhanced, protein_edge_attr_enhanced], dim=0)

    # Metadata for tracking the origin of each edge, not used in training/inference
    origin_edges = torch.cat([
        torch.zeros(ligand.edge_index.size(1),  dtype=torch.long),
        torch.ones (protein.edge_index.size(1), dtype=torch.long) 
    ], dim=0)

    # ------------------------------------------------------------------ #
    # 3. (optional) ligand-protein cross edges (distance < r_cut)    
    # ------------------------------------------------------------------ #
    if connect_cross:
        # Compute per-atom pharmacophore flags (only needed for cross-edge interactions)
        lig_flags = _atom_flags_pharmacophore(ligand.mol)
        pro_flags = _atom_flags_pharmacophore(protein.mol)
        
        # all-pairs distances between ligand & protein atoms
        d = torch.cdist(pos[:num_lig], pos[num_lig:])      # (num_lig, num_pro)
        # pairs within cutoff
        src, dst = torch.nonzero(d < r_cut, as_tuple=True)  # ligand idx, protein idx (0-based within their own sets)

        # map to global indices
        src_idx = src                               # [E_cross]
        dst_idx = dst + prot_shift                  # [E_cross]

        cross_edge_index = torch.stack([src_idx, dst_idx], dim=0)  # [2, E_cross]
        rev_edge_index   = torch.flip(cross_edge_index, [0])       # undirected

        # distance feature (kept as in your code)
        dist = d[src, dst].unsqueeze(1)  # (E_cross, 1)

        # ------- interaction one-hots (booleans → float) -------
        # gather flags for selected pairs
        def _gather(flag_lig, flag_pro):
            # flag_* are [N] uint8 tensors
            return flag_lig[src].to(torch.bool), flag_pro[dst].to(torch.bool)

        lig_don, pro_acc = _gather(lig_flags["donor"],    pro_flags["acceptor"])
        lig_acc, pro_don = _gather(lig_flags["acceptor"], pro_flags["donor"])
        lig_cat, pro_ani = _gather(lig_flags["cationic"], pro_flags["anionic"])
        lig_ani, pro_cat = _gather(lig_flags["anionic"],  pro_flags["cationic"])
        lig_aro, pro_aro = _gather(lig_flags["aromatic"], pro_flags["aromatic"])
        lig_hyd, pro_hyd = _gather(lig_flags["hydrophobe"], pro_flags["hydrophobe"])

        # geometric thresholds (Å)
        hb_cut   = 3.5
        salt_cut = 4.0
        pi_cut   = 6.0
        catpi_cut= 6.0
        hyd_cut  = 4.5

        hbond = ((lig_don & pro_acc) | (lig_acc & pro_don)) & (dist.squeeze(1) <= hb_cut)
        salt_bridge = ((lig_cat & pro_ani) | (lig_ani & pro_cat)) & (dist.squeeze(1) <= salt_cut)
        pi_pi = (lig_aro & pro_aro) & (dist.squeeze(1) <= pi_cut)
        cation_pi = ((lig_cat & pro_aro) | (lig_aro & pro_cat)) & (dist.squeeze(1) <= catpi_cut)
        hydrophobe_contact = (lig_hyd & pro_hyd) & (dist.squeeze(1) <= hyd_cut)

        inter_feats = torch.stack([
            hbond, salt_bridge, pi_pi, cation_pi, hydrophobe_contact
        ], dim=1).to(torch.float)   # (E_cross, K_INTER)

        # ------- assemble cross-edge features -------
        # RBF for cross-distance using the SAME M_RBF and r_cut as your cross cutoff
        rbf_cross = rbf_expand(dist, r_cut=r_cut, M=M_RBF)  # (E_cross, M_RBF)

        # keep the first 9 "bond categorical" slots as zeros for cross edges
        base_zeros = torch.zeros(rbf_cross.size(0), 9, dtype=torch.float, device=dist.device)

        # cross base matches intra layout: [9 zeros] + [RBF(dist)]
        cross_attr_base = torch.cat([base_zeros, rbf_cross], dim=1)  # (E_cross, 9+M_RBF)

        # add interaction flags then type one-hot [0,0,1]
        cross_type = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float).repeat(dist.size(0), 1)

        # final: [25] + [K_INTER] + [3] = [25] + [5] + [3] = 33D
        cross_attr = torch.cat([cross_attr_base, inter_feats, cross_type], dim=1)

        # Sanity check shape matches intra edges
        if cross_attr.size(1) != edge_attr.size(1):
            # edge_attr currently holds intra edges (lig & pro) after enhancement
            raise ValueError(
                f"Cross edge feature dim {cross_attr.size(1)} != intra edge dim {edge_attr.size(1)}. "
                f"Expected 25 + {K_INTER} + 3 = {25 + K_INTER + 3}."
            )

        # concat both directions
        edge_index = torch.cat([edge_index, cross_edge_index, rev_edge_index], dim=1)
        edge_attr  = torch.cat([edge_attr,  cross_attr,       cross_attr],     dim=0)
        origin_edges = torch.cat([
            origin_edges,
            torch.full((cross_attr.size(0)*2,), 2, dtype=torch.long)   # 2 = cross
        ], dim=0)

    # ------------------------------------------------------------------ #
    # 4. pack into new Data object
    # ------------------------------------------------------------------ #
    merged = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        origin_nodes=origin_nodes,      # handy for masking later
        origin_edges=origin_edges,
        ligand_mol=ligand.mol,
        protein_mol=protein.mol
    )
    return merged

import traceback
from multiprocessing import get_context

def _atomic_torch_save(obj: object, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def build_and_save_pair(task):
        tuple_id, ligand_path, pocket_path, out_root = task
        try:
            lig_out = os.path.join(out_root, "ligand", f"{tuple_id}.pt")
            pro_out = os.path.join(out_root, "protein", f"{tuple_id}.pt")

            if os.path.exists(lig_out) and os.path.exists(pro_out):
                return tuple_id, "skip", ""

            ligand = process_ligand_sdf(ligand_path)
            protein = process_protein_pdb_ligand_style(pocket_path)

            ligand.id = tuple_id
            protein.id = tuple_id

            _atomic_torch_save(ligand.cpu(), lig_out)
            _atomic_torch_save(protein.cpu(), pro_out)
            return tuple_id, "ok", ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
            return tuple_id, "fail", err


def merge_from_precomputed(task):
    """Merge precomputed ligand and protein .pt graphs and save merged .pt

    task = (tuple_id, ligand_pt_path, protein_pt_path, merged_out_path, connect_cross, r_cut, force_reload)
    """
    tuple_id, lig_pt, pro_pt, out_path, connect_cross, r_cut, force_reload = task
    try:
        # Skip if target exists and loads
        if os.path.exists(out_path) and not force_reload:
            try:
                g = torch.load(out_path, map_location="cpu", weights_only=False)
                if getattr(g, "pos", None) is not None and getattr(g, "edge_index", None) is not None:
                    return tuple_id, "skip", ""
            except Exception:
                pass

        lig = torch.load(lig_pt, map_location="cpu", weights_only=False)
        pro = torch.load(pro_pt, map_location="cpu", weights_only=False)

        merged = merge_ligand_and_protein(lig, pro, connect_cross=connect_cross, r_cut=r_cut)
        merged.id = tuple_id
        _atomic_torch_save(merged, out_path)
        return tuple_id, "ok", ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        return tuple_id, "fail", err


def worker_init() -> None:
    """Initializer for worker processes to set thread/env limits safely under spawn."""
    import os as _os
    import torch as _torch
    _os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "TBB_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
    })
    try:
        _torch.set_num_threads(1)
        _torch.set_num_interop_threads(1)
    except Exception:
        pass

def create_graphs_from_dataset(
                df: pd.DataFrame, 
                preprocessed_graphs_path: str, 
                merged_data_path_root: str,
                connect_cross: bool,
                r_cut: float,
                force_reload: bool = False,
                num_workers: int = 1,
                chunksize: int = 8,
                max_tasks_per_child: int = 10,
            ): 
    """
    Process:
    1. Reads graphs from ligand/protein directory 
    2. Creates merged graphs
    3. Stores them in merged_data_path_root
    """
    logger = logging.getLogger(__name__)

    # ids from dataframe
    t = df['Target ChEMBLID'].astype(str).str.strip()
    l = df['Molecule ChEMBLID'].astype(str).str.strip()
    df_ids = set((t + '_' + l).unique())

    # seperate graph dirs checks
    lig_dir = os.path.join(preprocessed_graphs_path, "ligand")
    pro_dir = os.path.join(preprocessed_graphs_path, "protein")

    lig_files = [f for f in os.listdir(lig_dir) if f.endswith(".pt")]
    pro_files = [f for f in os.listdir(pro_dir) if f.endswith(".pt")]

    # Extract IDs from file names
    lig_ids = {os.path.splitext(f)[0] for f in lig_files}
    pro_ids = {os.path.splitext(f)[0] for f in pro_files}
    
    # Find intersection - only process files that exist in both directories
    common_ids = lig_ids & pro_ids
    
    # Log file count information
    logger.info(f"Found {len(lig_files)} ligand files, {len(pro_files)} protein files")
    logger.info(f"Common IDs (files in both directories): {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise RuntimeError("No common files found between ligand and protein directories")
    
    # Log missing files for information (but don't fail)
    only_lig = sorted(lig_ids - pro_ids)
    only_pro = sorted(pro_ids - lig_ids)
    if only_lig:
        logger.warning(f"Ligand files without matching protein files: {len(only_lig)} (showing first 10: {only_lig[:10]})")
    if only_pro:
        logger.warning(f"Protein files without matching ligand files: {len(only_pro)} (showing first 10: {only_pro[:10]})")

    # Print graph creation config for info
    logger.info(f"Cross connect: {connect_cross}, r_cut: {r_cut}")
    
    tasks = []
    # Use only files that exist in both directories and are also in the dataframe
    ids = sorted(common_ids & df_ids)
    logger.info(f"IDs to process (in both directories AND dataframe): {len(ids)}")
    for tid in tqdm(ids, total=len(ids), desc="Build merge tasks", file=sys.stdout):
        lig_pt = os.path.join(lig_dir, f"{tid}.pt")
        pro_pt = os.path.join(pro_dir, f"{tid}.pt")
        out_pt = os.path.join(merged_data_path_root, f"{tid}.pt")
        tasks.append((tid, lig_pt, pro_pt, out_pt, connect_cross, r_cut, force_reload))

    ok = skipped = failed = 0

    if num_workers == 1:
        print("Creating merged graphs sequentially")
        for task in tqdm(tasks, total=len(tasks), file=sys.stdout,
                         mininterval=0.2, smoothing=0, dynamic_ncols=True,
                         desc="Merging graphs (sequential)"):
            # merge_from_precomputed is called the same way as in pool.map (single tuple arg)
            tid, status, msg = merge_from_precomputed(task)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                logger.error(f"{tid} failed: {msg}")
            sys.stdout.flush()
            sys.stderr.flush()
    else:
        print(f"Creating merged graphs parallel with {num_workers} workers")
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=worker_init,
            maxtasksperchild=max_tasks_per_child,
        ) as pool:
    
            iterator = pool.imap_unordered(merge_from_precomputed, tasks, chunksize=chunksize)
    
            for tid, status, msg in tqdm(
                iterator,
                total=len(tasks),
                file=sys.stdout,
                mininterval=0.2,
                smoothing=0,
                dynamic_ncols=True,
            ):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    logger.error(f"{tid} failed: {msg}")
                sys.stdout.flush()
                sys.stderr.flush()
    
    logger.info(f"Finished. ok={ok}, skipped={skipped}, failed={failed}, total={len(tasks)}")
    logger.info("Merged graphs creation finished.")
    

def create_single_graphs(index_path, out_root):
    os.makedirs(out_root, exist_ok=True)
    lig_dir = os.path.join(out_root, "ligand")
    pro_dir = os.path.join(out_root, "protein")
    os.makedirs(lig_dir, exist_ok=True)
    os.makedirs(pro_dir, exist_ok=True)

    df = pd.read_csv(index_path)
    tasks = []
    for idx, row in tqdm(df.iterrows(), total=len(df), file=sys.stdout):
        tuple_id = row['Target ChEMBLID'] + "_" + row['Molecule ChEMBLID']
        ligand_path = row['ligand_sdf_path']
        pocket_path = row['pocket_pdb_path']
        tasks.append((tuple_id, ligand_path, pocket_path, out_root))
    
    ok = skipped = failed = 0
    for task in tqdm(tasks, total=len(tasks), file=sys.stdout,
                        mininterval=0.2, smoothing=0, dynamic_ncols=True,
                        desc="Creating single graphs"):

        tid, status, msg = build_and_save_pair(task)


        if status == "ok":
            ok += 1
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
            logger.error(f"{tid} failed: {msg}")
        sys.stdout.flush()
        sys.stderr.flush()
    
    logger.info(f"Finished. ok={ok}, skipped={skipped}, failed={failed}, total={len(tasks)}")
    logger.info("Program finished.") 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", 
        type=str, 
        #default="/data2/PDBBind/processed/indexes/Index_pdbbind_general.csv"
        default="/rds/general/user/kgb24/ephemeral/BindingNetv2/processed/indexes/Index_pretrain_soft_overlap_tanimoto_proteins.csv"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "--chunksize", 
        type=int, 
        default=8
    )
    parser.add_argument(
        "--phase",
        choices=["build", "merge"],
        default="build",
        help="Phase to run: build ligand/protein or merge precomputed",
    )
    parser.add_argument(
        "--max_tasks_per_child",
        type=int,
        default=10,
        help="Recycle worker processes after this many tasks to avoid hangs/leaks",
    )
    parser.add_argument(
        "--out_root", 
        type=str,
        #default="/data2/PDBBind/processed/etnn/base_graphs_simple"
        #default="/data2/PDBBind/processed/etnn/casf_graphs_simple"
        default="/rds/general/user/kgb24/ephemeral/BindingNetv2/processed/pretrain_graphs"
    )
    parser.add_argument(
        "--connect_cross",
        action="store_true",
        help="If set in merge phase, create ligand-protein cross edges within r_cut",
    )
    parser.add_argument(
        "--r_cut",
        type=float,
        default=5.0,
        help="Distance cutoff for cross edges in merge phase",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/rds/general/user/kgb24/home/topological-equivariant-networks/logs/pretrain_graphs.log"
    )
    args = parser.parse_args()

    # Logger writing only from main process
    log_path = args.log_path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    os.makedirs(args.out_root, exist_ok=True)
    lig_dir = os.path.join(args.out_root, "ligand")
    pro_dir = os.path.join(args.out_root, "protein")
    os.makedirs(lig_dir, exist_ok=True)
    os.makedirs(pro_dir, exist_ok=True)

    tasks = []
    if args.phase == "build":
        df = pd.read_csv(args.index)
        for idx, row in tqdm(df.iterrows(), total=len(df), file=sys.stdout):
            tuple_id = row['Target ChEMBLID'] + "_" + row['Molecule ChEMBLID']
            ligand_path = row['ligand_sdf_path']
            pocket_path = row['pocket_pdb_path']
            tasks.append((tuple_id, ligand_path, pocket_path, args.out_root))
    else:
        # Phase 2: build tasks from precomputed ligand/protein directories
        lig_dir = os.path.join(args.out_root, "ligand")
        pro_dir = os.path.join(args.out_root, "protein")
        out_dir = os.path.join(args.out_root, "merged")
        os.makedirs(out_dir, exist_ok=True)
        lig_files = [f for f in os.listdir(lig_dir) if f.endswith(".pt")]
        pro_files = [f for f in os.listdir(pro_dir) if f.endswith(".pt")]

        # Basic length check
        if len(lig_files) != len(pro_files):
            raise RuntimeError(
                f"Mismatch between ligand/protein file counts: ligand={len(lig_files)} vs protein={len(pro_files)}"
            )

        # Stronger: ID set equality check
        lig_ids = {os.path.splitext(f)[0] for f in lig_files}
        pro_ids = {os.path.splitext(f)[0] for f in pro_files}
        if lig_ids != pro_ids:
            only_lig = sorted(lig_ids - pro_ids)
            only_pro = sorted(pro_ids - lig_ids)
            msg = ["Ligand/Protein ID sets differ even though counts match."]
            if only_lig:
                msg.append(f"Only in ligand (showing up to 20): {only_lig[:20]}")
            if only_pro:
                msg.append(f"Only in protein (showing up to 20): {only_pro[:20]}")
            raise RuntimeError(" \n".join(msg))
            
        ids = sorted(lig_ids.intersection(pro_ids))
        for tid in tqdm(ids, total=len(ids), desc="Build merge tasks", file=sys.stdout):
            lig_pt = os.path.join(lig_dir, f"{tid}.pt")
            pro_pt = os.path.join(pro_dir, f"{tid}.pt")
            out_pt = os.path.join(out_dir, f"{tid}.pt")
            tasks.append((tid, lig_pt, pro_pt, out_pt, args.connect_cross, args.r_cut))
    
    ok = skipped = failed = 0
    if args.num_workers == 1:
        print("Creating graphs sequentially")
        for task in tqdm(tasks, total=len(tasks), file=sys.stdout,
                         mininterval=0.2, smoothing=0, dynamic_ncols=True,
                         desc=f"Creating graphs (sequential), phase={args.phase}"):
            # merge_from_precomputed is called the same way as in pool.map (single tuple arg)
            if args.phase == "build":
                tid, status, msg = build_and_save_pair(task)
            else:
                tid, status, msg = merge_from_precomputed(task)

            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                logger.error(f"{tid} failed: {msg}")
            sys.stdout.flush()
            sys.stderr.flush()
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=args.num_workers,
            initializer=worker_init,
            maxtasksperchild=args.max_tasks_per_child,
        ) as pool:
            if args.phase == "build":
                iterator = pool.imap_unordered(build_and_save_pair, tasks, chunksize=args.chunksize)
            else:
                iterator = pool.imap_unordered(merge_from_precomputed, tasks, chunksize=args.chunksize)

            for tid, status, msg in tqdm(
                iterator,
                total=len(tasks),
                file=sys.stdout,
                mininterval=0.2,
                smoothing=0,
                dynamic_ncols=True,
            ):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    logger.error(f"{tid} failed: {msg}")
                sys.stdout.flush()
                sys.stderr.flush()
    
    logger.info(f"Finished. ok={ok}, skipped={skipped}, failed={failed}, total={len(tasks)}")
    logger.info("Program finished.")



