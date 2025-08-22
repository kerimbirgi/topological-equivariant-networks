import os
import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType as BT
import argparse
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import sys
#from rdkit.Chem import rdFreeSASA
#from rdkit.Chem import Crippen

logger = logging.getLogger(__name__)

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

# Expected feature dimensions for validation
EXPECTED_NODE_FEATURES = 13  # 6 basic atom properties + 7 hybridization one-hot
EXPECTED_EDGE_FEATURES = 10  # 7 bond types + is_conj + in_ring + length



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
        in_ring = [1 if bond.IsInRing() else 0]

        p1, p2 = mol.GetConformer().GetAtomPosition(start), mol.GetConformer().GetAtomPosition(end)
        length = [p1.Distance(p2)]

        feats = bt_oh + is_conj + in_ring + length
        edge_feat_list += [feats, feats]
        

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
        ring_flag = [int(bond.IsInRing())]
        p1, p2 = conf.GetAtomPosition(start), conf.GetAtomPosition(end)
        length = [p1.Distance(p2)]
        feats = bt_oh + is_conj + ring_flag + length
        edge_feat_list += [feats, feats]

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
    # Assert required tensors exist (helps type checker and avoids runtime None errors)
    assert ligand.x is not None and ligand.pos is not None and ligand.edge_index is not None and ligand.edge_attr is not None
    assert protein.x is not None and protein.pos is not None and protein.edge_index is not None and protein.edge_attr is not None

    # number of ligand and protein atoms
    num_lig = ligand.x.size(0) 
    num_pro = protein.x.size(0) 

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
    else:
        # [intra_ligand, intra_protein]
        lig_type = torch.tensor([[1, 0]], dtype=torch.float).repeat(num_lig_edges, 1) # (num_lig_edges, 2)
        pro_type = torch.tensor([[0, 1]], dtype=torch.float).repeat(num_pro_edges, 1) # (num_pro_edges, 2)
    
    # Concatenate original edge features with type features
    ligand_edge_attr_enhanced = torch.cat([ligand.edge_attr, lig_type], dim=1)
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
        # all-pairs distances between ligand & protein atoms
        d = torch.cdist(pos[:num_lig], pos[num_lig:])      # (num_lig, num_pro)
        src, dst = torch.nonzero(d < r_cut, as_tuple=True) # Find pairs within cutoff distance
        
        # map back to global indices, shift protein indices by num_lig
        src_idx = src
        dst_idx = dst + prot_shift

        cross_edge_index = torch.stack([src_idx, dst_idx], dim=0)
        rev_edge_index   = torch.flip(cross_edge_index, [0])        # undirected

        # Cross-edge features: distance + edge type
        dist = d[src, dst][:, None]            # (E_cross,1)
        # Pad to match molecular bond feature order: bond_type(7) + conj(1) + ring(1) + distance(1)
        # Use -1 as sentinel value to clearly distinguish cross-connections from molecular bonds
        original_edge_feat_dim = ligand.edge_attr.size(1)
        cross_attr_base = torch.cat([
            torch.zeros(dist.size(0), original_edge_feat_dim - 1),  # -1 sentinel for non-bond features
            dist                                                           # Distance at the end, like molecular bonds
        ], dim=1)
        
        # Add edge type for cross edges: [0, 0, 1] = inter_molecular
        # (This only happens when connect_cross=True, so we always use 3-way encoding here)
        cross_type = torch.tensor([[0, 0, 1]], dtype=torch.float).repeat(cross_attr_base.size(0), 1)
        cross_attr = torch.cat([cross_attr_base, cross_type], dim=1)

        # Sanity check: dimensions of cross edge features should be same for protein and ligand
        if cross_attr.size(1) != edge_attr.size(1):
            raise ValueError(
                f"Cross edge feature dimension mismatch: "
                f"cross_attr={cross_attr.size(1)} vs existing edge_attr={edge_attr.size(1)}. "
                f"original_edge_feat_dim={original_edge_feat_dim}, "
                f"connect_cross={connect_cross}"
            )

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
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", 
        type=str, 
        default="/rds/general/user/kgb24/ephemeral/BindingNetv2/processed/indexes/Index_BindingNetv2_pockets.csv"
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
        default="/rds/general/user/kgb24/home/topological-equivariant-networks/data/bindingnetcc/base_graphs/preprocessed"
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
        default="/rds/general/user/kgb24/home/topological-equivariant-networks/logs/single_graphs_processing.log"
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
        print("Creating merged graphs sequentially")
        for task in tqdm(tasks, total=len(tasks), file=sys.stdout,
                         mininterval=0.2, smoothing=0, dynamic_ncols=True,
                         desc="Merging graphs (sequential)"):
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



