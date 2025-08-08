import hydra
from omegaconf import DictConfig, OmegaConf
from etnn.bindingnet.bindingnetcc import BindingNetCC
import utils
import os
import pandas as pd

def visualize_3d(pos, edge_index, origin_nodes, num_nodes):
    """Create interactive 3D visualization with Plotly"""
    try:
        import plotly.graph_objects as go
        import numpy as np
        print("Plotly imported successfully, creating 3D plot...")
        
        xyz = pos.numpy()
        edges = edge_index.t().numpy()
        print(f"Plotting {xyz.shape[0]} nodes and {edges.shape[0]} edges")
        
        # Build edge segments for plotting
        xe, ye, ze = [], [], []
        for s, t in edges:
            xe += [xyz[s,0], xyz[t,0], None]
            ye += [xyz[s,1], xyz[t,1], None] 
            ze += [xyz[s,2], xyz[t,2], None]
        
        # Color nodes by origin (ligand=blue, protein=red)
        node_colors = ['blue' if origin == 0 else 'red' for origin in origin_nodes]
        
        edge_trace = go.Scatter3d(
            x=xe, y=ye, z=ze,
            mode="lines",
            line=dict(width=2, color="gray"),
            name="Edges"
        )
        
        node_trace = go.Scatter3d(
            x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
            mode="markers",
            marker=dict(size=4, color=node_colors),
            name="Atoms",
            text=[f"Node {i}: {'Ligand' if origin_nodes[i]==0 else 'Protein'}" 
                  for i in range(num_nodes)],
            hovertemplate="%{text}<extra></extra>"
        )
        
        fig = go.Figure([edge_trace, node_trace])
        fig.update_layout(
            title="Ligand-Protein Complex (Blue=Ligand, Red=Protein)",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)", 
                zaxis_title="Z (Å)"
            )
        )
        
        print("Attempting to show 3D plot...")
        try:
            # Try different rendering options
            fig.show(renderer="browser")  # Force browser rendering
        except Exception as e:
            print(f"Browser rendering failed: {e}")
            try:
                # Save as HTML file instead
                fig.write_html("ligand_protein_3d.html")
                print("3D plot saved as 'ligand_protein_3d.html' - open in browser")
            except Exception as e2:
                print(f"HTML export also failed: {e2}")
        
    except ImportError as e:
        print(f"Install plotly for 3D visualization: pip install plotly")
        print(f"Import error: {e}")
    except Exception as e:
        print(f"3D visualization error: {e}")
        import traceback
        traceback.print_exc()


def visualize_2d(pos, edge_index, origin_nodes, num_nodes):
    """Create 2D visualization with matplotlib and NetworkX"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create networkx graph manually
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_list = [(int(edge_index[0, i]), int(edge_index[1, i])) 
                     for i in range(edge_index.shape[1])]
        G.add_edges_from(edge_list)
        
        # Use 2D projection of 3D coordinates
        pos_2d = {i: pos[i, :2].tolist() for i in range(num_nodes)}
        
        # Color by origin
        node_colors = ['blue' if origin_nodes[i] == 0 else 'red' 
                      for i in range(num_nodes)]
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos=pos_2d, node_color=node_colors, node_size=20, 
                edge_color="gray", alpha=0.7, width=0.5)
        plt.title("2D Projection: Blue=Ligand, Red=Protein")
        plt.axis('equal')
        plt.show()
        
    except ImportError:
        print("Install matplotlib and networkx for 2D visualization: pip install matplotlib networkx")

def test_etnn_forward_pass(model, cc_data):
    """Test a forward pass with an ETNN model to ensure compatibility"""
    try:
        import torch
        print("\n=== Testing ETNN Forward Pass ===")
    
        # Put model in eval mode
        model.eval()
        
        # Perform forward pass
        with torch.no_grad():
            print("Performing forward pass...")
    
            output = model(cc_data)

            print(f"Forward pass successful!")
            print(f"Output type: {type(output)}")

    except ImportError as e:
        print(f"Missing dependencies for ETNN test: {e}")
        print("This is expected if ETNN model classes aren't properly imported")
    except Exception as e:
        print(f"ETNN forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis might be expected - the model may need specific combinatorial complex format")


def visualize_graph(g):
    # Visualize the merged graph
    print(f"Data type: {type(g)}")
    
    # Handle both regular Data objects and dictionary format from lifter
    if isinstance(g, dict):
        # Dictionary format from combinatorial complex lifter
        print(f"Dictionary keys: {list(g.keys())}")
        print(f"Graph has {g['x'].shape[0]} nodes and {g['edge_index'].shape[1]} edges")
        print(f"Node features shape: {g['x'].shape}")
        print(f"Edge features shape: {g['edge_attr'].shape}")
        print(f"Position shape: {g['pos'].shape}")
        
        # Extract data for visualization
        pos = g['pos']
        edge_index = g['edge_index']
        origin_nodes = g['origin_nodes']
        num_nodes = g['x'].shape[0]
    else:
        # Regular PyG Data object
        print(f"Graph has {g.num_nodes} nodes and {g.num_edges} edges")
        print(f"Node features shape: {g.x.shape}")
        print(f"Edge features shape: {g.edge_attr.shape}")
        print(f"Position shape: {g.pos.shape}")
        
        # Extract data for visualization
        pos = g.pos
        edge_index = g.edge_index
        origin_nodes = g.origin_nodes
        num_nodes = g.num_nodes

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
            'Ligand ChEMBLID': ['TEST_LIGAND'],
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
    test_etnn_forward_pass(model, ds[0])
    
    # uncomment to visualize the graph
    # visualize_graph(ds[0])

if __name__ == "__main__":
    """Quick functional test using sample files in test_data/"""
    main()

    
    
    
    



