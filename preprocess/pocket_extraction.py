import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO, Select
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import pandas as pd
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def extract_batch(batch):
    # batch is a list of (protein_pdb, ligand_sdf, cutoff) tuples
    # returns pocket paths in that same tuple order
    return [str(extract_pocket(p, l, c)) for p, l, c in batch]

class PocketSelect(Select):
        def __init__(self, coords, cutoff=5.0):
            self.coords = coords
            self.cutoff = cutoff

        def accept_atom(self, atom):
            dists = np.linalg.norm(self.coords - atom.get_coord(), axis=1)
            return bool((dists <= self.cutoff).any())

def extract_pocket(protein_pdb: Path, ligand_sdf: Path, cutoff: float):
    #tqdm.write(f"[{threading.current_thread().name}] processing {str(protein_pdb)}")
    output = protein_pdb.parent / f"pocket_{int(cutoff)}A.pdb"
    if output.exists():
        #tqdm.write(f"[{threading.current_thread().name}] output exists!")
        return output
    ligands = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    ligand = ligands[0]
    conf = ligand.GetConformer()
    lig_coords = np.array(conf.GetPositions())  # shape (N_atoms, 3)

    parser = PDBParser()
    structure = parser.get_structure('protein', protein_pdb)

    selector = PocketSelect(lig_coords, cutoff=cutoff)
    io = PDBIO()
    io.set_structure(structure)
    
    protein_pdb = protein_pdb
    io.save(str(output), selector)
    #tqdm.write(f"[{threading.current_thread().name}] processing done!")
    return output

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("pocket_extraction.log", mode="w"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, 
                        default="/data2/BindingNetv2/processed/indexes/Index_processed_bindingnetv2.csv")
    parser.add_argument("--out_index", type=Path,
                        default="/data2/BindingNetv2/processed/indexes/Index_bindingnetv2_pockets.csv")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    df = pd.read_csv(args.index)
    pocket_paths = [] # pocket output paths

    if args.workers > 1:
        print(f"Extracting pockets with {args.workers} workers")

        #prots, ligs, cuts = [], [], []
        #for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating parallel tasks"):
        #    prots.append(Path(row['protein_pdb_path']))
        #    ligs.append(Path(row['ligand_sdf_path']))
        #    cuts.append(args.cutoff)
        prots = [Path(p) for p in df['protein_pdb_path'].tolist()]
        ligs  = [Path(l) for l in df['ligand_sdf_path'].tolist()]
        cuts  = [args.cutoff] * len(df)
        # Pre‐allocate the result list
        pocket_paths = [""] * len(df)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # 1) submit inside the context
            futures = {
                pool.submit(extract_pocket, p, l, c): idx
                for idx, (p, l, c) in enumerate(zip(prots, ligs, cuts))
            }

            # 2) use tqdm.write in the worker
            # (you've already added those calls in extract_pocket)

            # 3) consume with as_completed and update bar
            with tqdm(total=len(futures), desc="Extracting pockets") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    out = fut.result()
                    pocket_paths[idx] = str(out)
                    pbar.update(1)
                    #print(f"Finished {out.name}")  # won't clobber the bar
    else:
        print(f"Extracting pockets with 1 worker")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting pockets"):
            protein_id = row['Target ChEMBLID']
            ligand_id = row['Molecule ChEMBLID']
            protein_pdb = row['protein_pdb_path']
            ligand_sdf = row['ligand_sdf_path']
            if pd.isna(protein_pdb) or pd.isna(ligand_sdf):
                logger.warning(f"Skipping {protein_id}_{ligand_id} because protein_pdb or ligand_sdf is None")
                pocket_paths.append("")   # keep alignment
                continue
            output = extract_pocket(Path(protein_pdb), Path(ligand_sdf), args.cutoff)
            pocket_paths.append(str(output))
    
    df['pocket_pdb_path'] = pocket_paths
    df.to_csv(args.out_index, index=False)
    logger.info(f"Wrote updated index with pocket paths to {args.out_index}")
            