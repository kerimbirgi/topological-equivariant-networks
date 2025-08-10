import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.utils import one_hot
#from rdkit.Chem import rdFreeSASA
#from rdkit.Chem import Crippen
import os

"""
# Build feature factory once
fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef)

def donor_acceptor_masks(mol):
    donors, acceptors = set(), set()
    for f in feat_factory.GetFeaturesForMol(mol):
        if f.GetFamily() == "Donor":
            donors.update(f.GetAtomIds())
        elif f.GetFamily() == "Acceptor":
            acceptors.update(f.GetAtomIds())
    return donors, acceptors
"""

HYB_TYPES = ['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2']
#BOND_TYPES  = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
# Bond type to index mapping (used for one-hot)
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}



def process_ligand_sdf(sdf_path: str) -> Data:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    if len(supplier) > 1:
        raise IndexError(f"Unexpexcted number of Molecules in ligand.sdf: {sdf_path}")
    mol = supplier[0] # We process single molecule ligands, no disconnected ones.
    if mol is None:
        raise ValueError(f"Could not parse ligand SDF file: {sdf_path}")
    if not mol.GetConformer().Is3D():
        raise ValueError("Ligand is not 3D")

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

        atom_features.append(
            [num, deg, chg, mass, in_ring, aro] 
            + hyb_oh 
        )
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

        # Bond-type one-hot
        bt_idx  = BOND_TYPES[bond.GetBondType()]
        bt_oh   = [int(i == bt_idx) for i in range(len(BOND_TYPES))]
        #edge_types += 2 * [BOND_TYPES[bond.GetBondType()]]

        # conjugation flag
        is_conj = [1 if bond.GetIsConjugated() else 0]

        # ring edge flag
        in_ring = [1 if bond.IsInRing() else 0]

        # Maybe add euclidean distance?
        p1, p2 = mol.GetConformer().GetAtomPosition(start), mol.GetConformer().GetAtomPosition(end)
        length = [p1.Distance(p2)]

        feats = bt_oh + is_conj + in_ring + length
        edge_feat_list += [feats, feats]
        

    edge_index = torch.tensor([rows, cols], dtype=torch.long)  # [2, E]
    edge_attr  = torch.tensor(edge_feat_list, dtype=torch.float32)
    #edge_attr  = one_hot(edge_type, num_classes=len(BOND_TYPES)).to(torch.float)

    perm = (edge_index[0] * mol.GetNumAtoms() + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr  = edge_attr[perm]

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

def process_protein_pdb_ligand_style(pdb_path: str) -> Data:
    """Return a Data object with the SAME node & edge feature format
    used by *process_ligand_sdf* so that ligands and protein pockets can be
    merged into a single batch.
    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit failed to parse PDB: {pdb_path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass  # best-effort sanitisation – proteins often have unusual residues

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
        bt_idx = BOND_TYPES.get(bond.GetBondType(), 0)
        bt_oh  = [int(i == bt_idx) for i in range(len(BOND_TYPES))]
        is_conj = [int(bond.GetIsConjugated())]
        ring_flag = [int(bond.IsInRing())]
        p1, p2 = conf.GetAtomPosition(start), conf.GetAtomPosition(end)
        length = [p1.Distance(p2)]
        feats = bt_oh + is_conj + ring_flag + length
        edge_feat_list += [feats, feats]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(edge_feat_list, dtype=torch.float32)

    perm = (edge_index[0] * mol.GetNumAtoms() + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr  = edge_attr[perm]

    pos = torch.tensor([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=torch.float32)

    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)


def merge_ligand_and_protein(
        ligand:  Data,
        protein: Data,
        connect_cross: bool = False,
        r_cut: float = 5.0,
) -> Data:
    """
    Return a single Data object that contains both graphs.

    New attributes
    --------------
    origin_nodes : (N,)   0 = ligand node,   1 = protein node
    origin_edges : (E,)   0 = ligand edge,   1 = protein edge, 2 = cross edge
    """
    # ------------------------------------------------------------------ #
    # 1. nodes
    # ------------------------------------------------------------------ #
    num_lig = ligand.x.size(0)
    num_pro = protein.x.size(0)

    x   = torch.cat([ligand.x, protein.x], dim=0)          # (N, F)
    pos = torch.cat([ligand.pos, protein.pos], dim=0)      # (N, 3)

    # Atom-level origin flag: 0 = ligand, 1 = protein
    lig_flag = torch.zeros((num_lig, 1), dtype=x.dtype, device=x.device)
    pro_flag = torch.ones((num_pro, 1),  dtype=x.dtype, device=x.device)
    origin_flag = torch.cat([lig_flag, pro_flag], dim=0)   # (N, 1)
    x = torch.cat([x, origin_flag], dim=1)                 # (N, F+1)

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

    edge_index = torch.cat([edge_index_lig, edge_index_pro], dim=1)
    edge_attr  = torch.cat([ligand.edge_attr, protein.edge_attr], dim=0)

    origin_edges = torch.cat([
        torch.zeros(ligand.edge_index.size(1),  dtype=torch.long),  # ligand
        torch.ones (protein.edge_index.size(1), dtype=torch.long)   # protein
    ], dim=0)

    # ------------------------------------------------------------------ #
    # 3. optional ligand-protein cross edges (distance < r_cut)
    # ------------------------------------------------------------------ #
    if connect_cross:
        # all-pairs distances between ligand & protein atoms
        d = torch.cdist(pos[:num_lig], pos[num_lig:])      # (num_lig, num_pro)
        src, dst = torch.nonzero(d < r_cut, as_tuple=True)
        # map back to global indices
        src_idx = src
        dst_idx = dst + prot_shift

        cross_edge_index = torch.stack([src_idx, dst_idx], dim=0)
        rev_edge_index   = torch.flip(cross_edge_index, [0])        # undirected

        # very simple cross-edge feature: distance only (1-dim)
        dist = d[src, dst][:, None]            # (E_cross,1)
        cross_attr = dist.repeat(1, edge_attr.size(1))     # pad to same dim

        edge_index = torch.cat([edge_index, cross_edge_index, rev_edge_index], dim=1)
        edge_attr  = torch.cat([edge_attr,  cross_attr,     cross_attr],     dim=0)
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
        origin_edges=origin_edges
    )
    return merged