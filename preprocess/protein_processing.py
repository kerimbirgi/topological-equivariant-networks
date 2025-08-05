import torch
import pickle
import argparse
import pandas as pd
import numpy as np
import lmdb
from scipy.spatial import cKDTree
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch_geometric.data import Data
import logging

#HYB_TYPES = ['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2']
#BOND_TYPES  = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']


# Protein stuff
ATOMS = ['C', 'N', 'O', 'S', 'P', 'H', 'F', 'Cl', 'Br', 'I']
RES   = [
    'ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]

# -------------------------------------------------------------------------
# 2️⃣  Helper: radial-basis expansion of distances  (edge feature)
# -------------------------------------------------------------------------
def rbf(dist, D_min=0.0, D_max=8.0, n_kernels=6, gamma=10.0):
    """
    dist : (E,1) float32  – raw Euclidean distance in Å
    returns (E, n_kernels) float32  – smooth encoding
    """
    centers = torch.linspace(D_min, D_max, n_kernels, device=dist.device)
    return torch.exp(-gamma * (dist - centers) ** 2)

def one_hot(x, vocab):
    v = np.zeros(len(vocab), np.float32)
    if x in vocab: v[vocab.index(x)] = 1
    return v

def atom_feat(atom, res):
    a_one = one_hot(atom.element.strip(), ATOMS)
    r_one = one_hot(res.get_resname(), RES)
    is_bb = float(atom.get_id() in ('N','CA','C','O'))
    return np.concatenate([a_one, r_one, [is_bb]], dtype=np.float32)

def load_pocket(pdb_path: str):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("protein", pdb_path)
    coords, feats, residue_ids, atom_ids = [], [], [], []
    bfactors, charges = [], []
    for atom in s.get_atoms():
        if atom.element == 'H':            # skip hydrogens – GINE & EGGNet usually do
            continue
        
        res = atom.get_parent()
        feats.append(atom_feat(atom, res))

        bfactors.append(atom.get_bfactor())          # temperature factor
        charges.append(atom.get_occupancy() or 0.0)  # formal charge if encoded

        coords.append(atom.coord.astype(np.float32))
        residue_ids.append(id(res))        # Python id ≈ unique per residue
        atom_ids.append(atom.serial_number)

    bfactors = np.array(bfactors, np.float32)
    bfactors = (bfactors - bfactors.mean()) / (bfactors.std() + 1e-6)
    charges  = np.array(charges,  np.float32)

    extra    = np.stack([bfactors, charges], axis=1)      # (N,2)
    feats    = np.concatenate([np.stack(feats), extra], axis=1)

    #return (
    #    np.stack(coords),                  # (N,3)
    #    np.stack(feats),                   # (N,F)
    #    residue_ids,                       # len N
    #    atom_ids,
    #    s
    #)
    return (
        np.stack(coords),          # (N,3)
        feats,                     # now (N, 31 + 2) → 33 channels
        residue_ids,
        atom_ids,
        s
    )

def process_protein_pdb(pdb_path: str, r_cut: float = 6.0) -> Data:
    xyz, feats, *_ = load_pocket(pdb_path)

    tree = cKDTree(xyz)
    idx = tree.query_pairs(r_cut, output_type='ndarray')
    idx = np.vstack([idx, idx[:, ::-1]])          # undirected

    dist = np.linalg.norm(xyz[idx[:,0]] - xyz[idx[:,1]], axis=1, keepdims=True)
    is_cov = (dist[:, 0] < 1.9).astype(np.float32)[:, None]
    dist_t  = torch.from_numpy(dist)                 # (E,1) tensor
    rbf_enc = rbf(dist_t, n_kernels=6)               # (E,6)

    # final edge_attr = [distance, RBF(6), covalent_flag]  → (E, 1+6+1) = (E,8)
    edge_attr = torch.cat([dist_t, rbf_enc, torch.from_numpy(is_cov)], dim=1)


    
    data = Data(
        x=torch.from_numpy(feats).float(),           # (N,33)
        pos=torch.from_numpy(xyz).float(),
        edge_index=torch.as_tensor(idx.T, dtype=torch.long),
        edge_attr=edge_attr.float()                  # (E,8)
    )

    return data



    

