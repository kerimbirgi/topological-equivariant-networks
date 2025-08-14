import hydra
from omegaconf import DictConfig, OmegaConf
from etnn.bindingnet.bindingnetcc import BindingNetCC
import utils
import os
import pandas as pd
from torch_geometric.loader import DataLoader
import torch
from etnn.combinatorial_data import CombinatorialComplexData
from preprocess.single_graph_processing import (
    process_ligand_sdf,
    process_protein_pdb_ligand_style,
    merge_ligand_and_protein,
)

def visualize_3d(pos, edge_index, origin_nodes, num_nodes, edge_origin=None, title_extra: str = ""):
    """3D plot colored by origin: 0=ligand(blue), 1=protein(red)"""
    try:
        import plotly.graph_objects as go
        import numpy as np

        xyz = pos.cpu().numpy()
        edges = edge_index.t().cpu().numpy()

        # colors from origin_nodes
        if origin_nodes is None:
            labels = np.zeros(num_nodes, dtype=int)
        else:
            labels = origin_nodes.detach().cpu().numpy().astype(int).reshape(-1)
        node_colors = np.where(labels == 0, "blue", "red").tolist()

        # Edge coloring by origin, if provided: 0=lig, 1=pro, 2=cross
        def append_edge(seg, s, t):
            seg[0].extend([xyz[s, 0], xyz[t, 0], None])
            seg[1].extend([xyz[s, 1], xyz[t, 1], None])
            seg[2].extend([xyz[s, 2], xyz[t, 2], None])

        segs = {
            0: ([], [], []),  # ligand
            1: ([], [], []),  # protein
            2: ([], [], []),  # cross
            "all": ([], [], []),
        }
        if edge_origin is not None:
            eo = edge_origin.detach().cpu().numpy().astype(int).tolist()
            for (s, t), o in zip(edges, eo):
                append_edge(segs.get(o, segs["all"]), s, t)
        else:
            for s, t in edges:
                append_edge(segs["all"], s, t)

        traces = []
        if segs["all"][0]:
            traces.append(go.Scatter3d(x=segs["all"][0], y=segs["all"][1], z=segs["all"][2], mode="lines",
                                       line=dict(width=2, color="gray"), name="Edges"))
        if segs[0][0]:
            traces.append(go.Scatter3d(x=segs[0][0], y=segs[0][1], z=segs[0][2], mode="lines",
                                       line=dict(width=2, color="#7f7f7f"), name="Ligand edges"))
        if segs[1][0]:
            traces.append(go.Scatter3d(x=segs[1][0], y=segs[1][1], z=segs[1][2], mode="lines",
                                       line=dict(width=2, color="#bdbdbd"), name="Protein edges"))
        if segs[2][0]:
            traces.append(go.Scatter3d(x=segs[2][0], y=segs[2][1], z=segs[2][2], mode="lines",
                                       line=dict(width=3, color="green"), name="Cross edges"))
        node_trace = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                  mode="markers",
                                  marker=dict(size=4, color=node_colors),
                                  name="Atoms")

        fig = go.Figure(traces + [node_trace])
        title = "Ligand(blue) vs Protein(red)"
        if title_extra:
            title = f"{title} — {title_extra}"
        fig.update_layout(title=title,
                          scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        fig.show(renderer="browser")
    except Exception as e:
        print(f"3D visualization error: {e}")


def visualize_2d(pos, edge_index, origin_nodes, num_nodes, edge_origin=None, title_extra: str = ""):
    """2D plot colored by origin: 0=ligand(blue), 1=protein(red)"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        ei = edge_index.cpu().numpy()
        edges = [(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])]
        G.add_edges_from(edges)

        # colors from origin_nodes
        if origin_nodes is None:
            labels = [0] * num_nodes
        else:
            labels = origin_nodes.detach().cpu().numpy().astype(int).reshape(-1).tolist()
        node_colors = ["blue" if lbl == 0 else "red" for lbl in labels]

        pos_2d = {i: pos[i, :2].cpu().tolist() for i in range(num_nodes)}
        # Edge colors
        if edge_origin is not None:
            eo = edge_origin.detach().cpu().numpy().astype(int).tolist()
            edge_colors = []
            for idx, _ in enumerate(edges):
                o = eo[idx] if idx < len(eo) else -1
                edge_colors.append("green" if o == 2 else ("#7f7f7f" if o == 0 else "#bdbdbd"))
        else:
            edge_colors = "gray"

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos=pos_2d, node_color=node_colors, node_size=20,
                edge_color=edge_colors, alpha=0.7, width=0.5)
        title = "Ligand(blue) vs Protein(red)"
        if title_extra:
            title = f"{title} — {title_extra}"
        plt.title(title)
        plt.axis('equal')
        plt.show()
    except Exception as e:
        print(f"2D visualization error: {e}")

def test_etnn_forward_pass(model, ds):
    """Test a forward pass with an ETNN model to ensure compatibility"""
    try:
        
        print("\n=== Testing ETNN Forward Pass ===")
    
        # Put model in eval mode
        model.eval()

        # Get dataloader
        #dataloader = DataLoader(ds, batch_size=1, shuffle=False)
        batch = next(iter(DataLoader(ds, batch_size=1)))
        
        # Perform forward pass
        with torch.no_grad():
            print("Performing forward pass...")
    
            output = model(batch)

            print(f"Forward pass successful!")
            print(f"Output type: {type(output)}")
            print(f"Output keys: {output}")

    except ImportError as e:
        print(f"Missing dependencies for ETNN test: {e}")
        print("This is expected if ETNN model classes aren't properly imported")
    except Exception as e:
        print(f"ETNN forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis might be expected - the model may need specific combinatorial complex format")


def visualize_graph(g, connect_cross: bool | None = None, r_cut: float | None = None):
    title_extra = None
    if connect_cross is not None and r_cut is not None:
        title_extra = f"cross={connect_cross}, r_cut={r_cut}"
    if isinstance(g, CombinatorialComplexData):
        pos = getattr(g, "pos", None)
        if pos is None:
            print("Graph has no positions; cannot visualize.")
            return
        num_nodes = pos.size(0)

        # try to get labels from attribute, else from last column of x_0 (origin flag)
        labels = getattr(g, "origin_nodes", None)
        if labels is None and hasattr(g, "x_0") and g.x_0 is not None and g.x_0.numel() > 0:
            # assume last feature column is origin flag (0 ligand, 1 protein)
            labels = (g.x_0[:, -1] > 0.5).long()

        # pick a 0-0 adjacency if available
        adj_keys = [k for k in g.keys() if k.startswith("adj_0_0")]
        if adj_keys:
            key = sorted(adj_keys)[0]
            edge_index = getattr(g, key)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)

        # choose one: 2D or 3D
        #visualize_2d(pos, edge_index, labels, num_nodes, None, title_extra or "")
        visualize_3d(pos, edge_index, labels, num_nodes, None, title_extra or "")
    else:
        pos = getattr(g, "pos", None)
        edge_index = getattr(g, "edge_index", None)
        if pos is None or edge_index is None:
            print("Graph missing pos or edge_index; cannot visualize.")
            return
        num_nodes = pos.size(0)
        visualize_2d(pos, edge_index, getattr(g, "origin_nodes", None), num_nodes,
                     getattr(g, "origin_edges", None), title_extra or "")

@hydra.main(config_path="conf/conf_bindingnet", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Locate test_data directory relative to project root
    base_dir = os.path.dirname(__file__)
    #print(f"base_dir: {base_dir}")
    test_dir = os.path.join(base_dir, 'test_data')
    #print(f"test_dir: {test_dir}")

    # Create a minimal CSV index
    csv_path = os.path.join(test_dir, 'index_test.csv')
    if not os.path.exists(csv_path):
        df = pd.DataFrame({
            'Target ChEMBLID': ['TEST_TARGET'],
            'Molecule ChEMBLID': ['TEST_LIGAND'],
            'ligand_sdf_path': [os.path.join(test_dir, 'ligand.sdf')],
            'pocket_pdb_path': [os.path.join(test_dir, 'pocket_6A.pdb')],
            '-logAffi': [7.0],
        })
        df.to_csv(csv_path, index=False)
    else:
        # Ensure required target column exists
        df = pd.read_csv(csv_path)
        if '-logAffi' not in df.columns:
            df['-logAffi'] = 7.0
            df.to_csv(csv_path, index=False)

    # Output directory inside test_data
    out_root = os.path.join(test_dir, 'bindingnet_out')
    merged_dir = os.path.join(out_root, 'preprocessed', 'merged')
    os.makedirs(merged_dir, exist_ok=True)

    # Ensure the expected merged file exists for the tuple id
    CONNECT_CROSS = False
    R_CUT = 4.0
    tuple_id = 'TEST_TARGET_TEST_LIGAND'
    merged_path = os.path.join(merged_dir, f'{tuple_id}.pt')

    print("Processing...")
    lig_path = os.path.join(test_dir, 'ligand.sdf')
    poc_path = os.path.join(test_dir, 'pocket_6A.pdb')
    ligand = process_ligand_sdf(lig_path)
    protein = process_protein_pdb_ligand_style(poc_path)
    merged = merge_ligand_and_protein(
        ligand,
        protein,
        connect_cross=CONNECT_CROSS,
        r_cut=R_CUT,
    )
    merged.id = tuple_id
    torch.save(merged, merged_path)

    print("Getting dataset...")
    ds = BindingNetCC(
        index=csv_path,
        root=out_root,
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        connect_cross=CONNECT_CROSS,
        r_cut=R_CUT,
        force_reload=True,
    )

    print("Getting model...")
    model = utils.get_model(cfg, ds)

    print('Processed files:', ds.processed_paths)
    print('Number of merged graphs:', len(ds))
    
    # Test forward pass with ETNN model
    #test_etnn_forward_pass(model, ds)
    
    # visualize the first graph
    if len(ds) > 0:
        # Try to surface cross-edge flag and radius in the title
        try:
            from math import isnan
            visualize_graph(ds[0], connect_cross=CONNECT_CROSS, r_cut=R_CUT)
        except Exception:
            visualize_graph(ds[0])
    else:
        print("Dataset is empty; nothing to visualize.")

if __name__ == "__main__":
    """Quick functional test using sample files in test_data/"""
    main()

    
    
    
    



