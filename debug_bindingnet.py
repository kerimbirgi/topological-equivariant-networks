import hydra
from omegaconf import DictConfig, OmegaConf
from etnn.bindingnet.bindingnetcc import BindingNetCC
import utils
import os
import pandas as pd
from torch_geometric.loader import DataLoader
import torch
from etnn.combinatorial_data import CombinatorialComplexData

def visualize_3d(pos, edge_index, origin_nodes, num_nodes):
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

        xe, ye, ze = [], [], []
        for s, t in edges:
            xe += [xyz[s, 0], xyz[t, 0], None]
            ye += [xyz[s, 1], xyz[t, 1], None]
            ze += [xyz[s, 2], xyz[t, 2], None]

        edge_trace = go.Scatter3d(x=xe, y=ye, z=ze, mode="lines",
                                  line=dict(width=2, color="gray"), name="Edges")
        node_trace = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                  mode="markers",
                                  marker=dict(size=4, color=node_colors),
                                  name="Atoms")

        fig = go.Figure([edge_trace, node_trace])
        fig.update_layout(title="Ligand(blue) vs Protein(red)",
                          scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        fig.show(renderer="browser")
    except Exception as e:
        print(f"3D visualization error: {e}")


def visualize_2d(pos, edge_index, origin_nodes, num_nodes):
    """2D plot colored by origin: 0=ligand(blue), 1=protein(red)"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        ei = edge_index.cpu().numpy()
        G.add_edges_from([(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])])

        # colors from origin_nodes
        if origin_nodes is None:
            labels = [0] * num_nodes
        else:
            labels = origin_nodes.detach().cpu().numpy().astype(int).reshape(-1).tolist()
        node_colors = ["blue" if lbl == 0 else "red" for lbl in labels]

        pos_2d = {i: pos[i, :2].cpu().tolist() for i in range(num_nodes)}
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos=pos_2d, node_color=node_colors, node_size=20,
                edge_color="gray", alpha=0.7, width=0.5)
        plt.title("Ligand(blue) vs Protein(red)")
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


def visualize_graph(g):
    if isinstance(g, CombinatorialComplexData):
        pos = g.pos
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
        #visualize_2d(pos, edge_index, labels, num_nodes)
        visualize_3d(pos, edge_index, labels, num_nodes)
    else:
        visualize_2d(g.pos, g.edge_index, getattr(g, "origin_nodes", None), g.num_nodes)

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
        })
        df.to_csv(csv_path, index=False)

    # Output directory inside test_data
    out_root = os.path.join(test_dir, 'bindingnet_out')

    print("Getting dataset...")
    ds = BindingNetCC(
        index=csv_path,
        root=out_root,
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        connect_cross=True,
        r_cut=5.0,
        mode='merged',
        force_reload=True,
    )

    print("Getting model...")
    model = utils.get_model(cfg, ds)

    print('Processed files:', ds.processed_paths)
    print('Number of merged graphs:', len(ds))
    
    # Test forward pass with ETNN model
    #test_etnn_forward_pass(model, ds)
    
    # uncomment to visualize the graph
    visualize_graph(ds)

if __name__ == "__main__":
    """Quick functional test using sample files in test_data/"""
    main()

    
    
    
    



