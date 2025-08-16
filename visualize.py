import os
import argparse
import torch


def visualize_3d(pos, edge_index, labels=None, title="Graph", origin_edges=None):
    """
    Visualize a 3D graph with different colors for different edge types.
    
    This function only uses data from the final merged graph:
    - pos: node positions
    - edge_index: edge connectivity  
    - labels: node types (from origin_nodes)
    - origin_edges: edge types (0=ligand, 1=protein, 2=cross)
    """
    try:
        import plotly.graph_objects as go
        import numpy as np

        notebook_env = False
        try: # find out if in notebook environment
            from IPython.core.getipython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                notebook_env = True

                import plotly.offline as pyo
                pyo.init_notebook_mode(connected=True)
                print("using notebook visualization method")
        except:
            print("using browser visualization method")
            notebook_env = False

        xyz = pos.cpu().numpy()
        edges = edge_index.t().contiguous().cpu().numpy()

        if labels is None:
            node_colors = ["blue"] * xyz.shape[0]
        else:
            lab = labels.detach().cpu().numpy().astype(int).reshape(-1)
            node_colors = np.where(lab == 0, "blue", "red").tolist()

        # Create separate edge traces for different edge types
        traces = []
        
        if origin_edges is not None:
            # Convert origin_edges to numpy
            origin_edges_np = origin_edges.detach().cpu().numpy().astype(int)
            
            # Define colors and names for each edge type
            edge_type_info = {
                0: {"color": "blue", "name": "Ligand Edges", "width": 2},
                1: {"color": "red", "name": "Protein Edges", "width": 2}, 
                2: {"color": "green", "name": "Cross Edges", "width": 3}
            }
            
            # Group edges by type
            for edge_type, info in edge_type_info.items():
                type_mask = origin_edges_np == edge_type
                if np.any(type_mask):
                    type_edges = edges[type_mask]
                    
                    xe, ye, ze = [], [], []
                    for s, t in type_edges:
                        xe += [xyz[s, 0], xyz[t, 0], None]
                        ye += [xyz[s, 1], xyz[t, 1], None]
                        ze += [xyz[s, 2], xyz[t, 2], None]
                    
                    edge_trace = go.Scatter3d(
                        x=xe, y=ye, z=ze, mode="lines",
                        line=dict(width=info["width"], color=info["color"]), 
                        name=info["name"]
                    )
                    traces.append(edge_trace)
        else:
            # Fallback to original single-color edges if no origin_edges
            xe, ye, ze = [], [], []
            for s, t in edges:
                xe += [xyz[s, 0], xyz[t, 0], None]
                ye += [xyz[s, 1], xyz[t, 1], None]
                ze += [xyz[s, 2], xyz[t, 2], None]

            edge_trace = go.Scatter3d(x=xe, y=ye, z=ze, mode="lines",
                                      line=dict(width=2, color="gray"), name="Edges")
            traces.append(edge_trace)

        node_trace = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                  mode="markers",
                                  marker=dict(size=4, color=node_colors),
                                  name="Nodes")
        traces.append(node_trace)

        fig = go.Figure(traces)
        fig.update_layout(title=title, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        if notebook_env:
            try:
                import plotly.offline as pyo
                pyo.iplot(fig)
            except ImportError:
                fig.show()
        else:
            fig.show(renderer="broswer")
            
        
    except Exception as e:
        print(f"3D visualization error: {e}")


def visualize_2d(pos, edge_index, labels=None, title="Graph", origin_edges=None):
    """
    Visualize a 2D graph with different colors for different edge types.
    
    This function only uses data from the final merged graph:
    - pos: node positions
    - edge_index: edge connectivity  
    - labels: node types (from origin_nodes)
    - origin_edges: edge types (0=ligand, 1=protein, 2=cross)
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        num_nodes = pos.size(0)
        ei = edge_index.cpu().numpy()

        if labels is None:
            node_colors = ["blue"] * num_nodes
        else:
            lab = labels.detach().cpu().numpy().astype(int).reshape(-1)
            node_colors = ["blue" if v == 0 else "red" for v in lab]

        pos_2d = {i: pos[i, :2].cpu().tolist() for i in range(num_nodes)}
        plt.figure(figsize=(12, 10))

        if origin_edges is not None:
            # Convert origin_edges to numpy
            origin_edges_np = origin_edges.detach().cpu().numpy().astype(int)
            
            # Define colors and styles for each edge type
            edge_type_info = {
                0: {"color": "blue", "name": "Ligand Edges", "width": 1.0, "style": "-"},
                1: {"color": "red", "name": "Protein Edges", "width": 1.0, "style": "-"}, 
                2: {"color": "green", "name": "Cross Edges", "width": 2.0, "style": "--"}
            }
            
            # Draw edges by type
            legend_handles = []
            for edge_type, info in edge_type_info.items():
                type_mask = origin_edges_np == edge_type
                if np.any(type_mask):
                    # Create a graph with only edges of this type
                    G_type = nx.Graph()
                    G_type.add_nodes_from(range(num_nodes))
                    type_edges = [(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]) if type_mask[i]]
                    G_type.add_edges_from(type_edges)
                    
                    # Draw this edge type
                    nx.draw_networkx_edges(
                        G_type, pos=pos_2d, 
                        edge_color=info["color"], 
                        width=info["width"],
                        style=info["style"],
                        alpha=0.7
                    )
                    
                    # Create legend handle
                    from matplotlib.lines import Line2D
                    legend_handles.append(
                        Line2D([0], [0], color=info["color"], linewidth=info["width"], 
                               linestyle=info["style"], label=info["name"])
                    )
            
            # Draw nodes on top
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            nx.draw_networkx_nodes(G, pos=pos_2d, node_color=node_colors, node_size=30, alpha=0.8)
            
            # Add legend
            plt.legend(handles=legend_handles, loc='upper right')
            
        else:
            # Fallback to original single-color edges if no origin_edges
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from([(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])])
            
            nx.draw(G, pos=pos_2d, node_color=node_colors, node_size=20,
                    edge_color="gray", alpha=0.8, width=0.6)

        plt.title(title)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"2D visualization error: {e}")


HYB_TYPES = ["UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP3D", "SP3D2"]
BOND_TYPE_NAMES = ["single", "double", "triple", "aromatic", "oneandahalf", "twoandahalf", "unspecified"]


def describe_graph_features(g) -> None:
    try:
        num_nodes = int(g.num_nodes) if hasattr(g, "num_nodes") else g.pos.size(0)
        num_edges = int(g.edge_index.size(1)) if hasattr(g, "edge_index") else 0
        print("\n=== Graph summary ===")
        print(f"nodes: {num_nodes}")
        print(f"edges: {num_edges}")
        if hasattr(g, "x") and g.x is not None:
            print(f"x shape: {tuple(g.x.shape)}")
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            print(f"edge_attr shape: {tuple(g.edge_attr.shape)}")
        if hasattr(g, "pos") and g.pos is not None:
            print(f"pos shape: {tuple(g.pos.shape)}")

        # Node feature schema guess
        if getattr(g, "x", None) is not None:
            fdim = g.x.size(1)
            base_names = [
                "atomic_num",
                "degree",
                "formal_charge",
                "mass",
                "in_ring",
                "is_aromatic",
            ]
            hyb_names = [f"hyb_{h}" for h in HYB_TYPES]
            expected_ligand_dim = len(base_names) + len(hyb_names)  # 13
            if fdim == expected_ligand_dim:
                names = base_names + hyb_names
                print("node features (ligand/protein):")
                for i, n in enumerate(names):
                    print(f"  [{i:2d}] {n}")
            elif fdim == expected_ligand_dim + 1:
                names = base_names + hyb_names + ["origin_flag(0=lig,1=prot)"]
                print("node features (merged):")
                for i, n in enumerate(names):
                    print(f"  [{i:2d}] {n}")
            else:
                print(f"node feature dim {fdim} (schema unknown to helper)")

        # Edge feature schema guess
        if getattr(g, "edge_attr", None) is not None:
            edim = g.edge_attr.size(1)
            expected_edge_dim = len(BOND_TYPE_NAMES) + 2 + 1  # 7 + 1 + 1 + 1 = 10
            expected_edge_dim_with_type = expected_edge_dim + 2  # +2 for 2-way edge type
            expected_edge_dim_with_3way_type = expected_edge_dim + 3  # +3 for 3-way edge type
            
            if edim == expected_edge_dim:
                e_names = [f"bond_{n}" for n in BOND_TYPE_NAMES] + ["is_conjugated", "in_ring", "length"]
                print("edge features (basic):")
                for i, n in enumerate(e_names):
                    print(f"  [{i:2d}] {n}")
            elif edim == expected_edge_dim_with_type:
                e_names = ([f"bond_{n}" for n in BOND_TYPE_NAMES] + 
                          ["is_conjugated", "in_ring", "length"] +
                          ["edge_type_ligand", "edge_type_protein"])
                print("edge features (with 2-way edge type):")
                for i, n in enumerate(e_names):
                    print(f"  [{i:2d}] {n}")
            elif edim == expected_edge_dim_with_3way_type:
                e_names = ([f"bond_{n}" for n in BOND_TYPE_NAMES] + 
                          ["is_conjugated", "in_ring", "length"] +
                          ["edge_type_ligand", "edge_type_protein", "edge_type_cross"])
                print("edge features (with 3-way edge type):")
                for i, n in enumerate(e_names):
                    print(f"  [{i:2d}] {n}")
            else:
                print(f"edge feature dim {edim} (schema unknown to helper)")
        
        # Edge type information
        if getattr(g, "origin_edges", None) is not None:
            import numpy as np
            origin_edges_np = g.origin_edges.detach().cpu().numpy()
            edge_type_counts = {}
            edge_type_names = {0: "ligand", 1: "protein", 2: "cross"}
            
            for edge_type in [0, 1, 2]:
                count = np.sum(origin_edges_np == edge_type)
                if count > 0:
                    edge_type_counts[edge_type_names[edge_type]] = count
            
            print("edge type distribution:")
            for edge_type, count in edge_type_counts.items():
                print(f"  {edge_type}: {count} edges")
    except Exception as e:
        print(f"describe_graph_features error: {e}")


def infer_labels(g):
    # Use explicit origin_nodes if present (this is the correct way for merged graphs)
    if hasattr(g, "origin_nodes") and g.origin_nodes is not None:
        return g.origin_nodes
    
    # Fallback heuristic for graphs that don't have origin_nodes
    try:
        if getattr(g, "x", None) is not None and g.x.dim() == 2 and g.x.size(1) >= 1:
            last_col = g.x[:, -1]
            if last_col.dtype.is_floating_point:
                return (last_col > 0.5).long()
            if last_col.dtype.is_floating_point is False:
                return last_col.long()
    except Exception:
        pass
    return None

def visualize_pt_graph(path, mode):
    if mode not in ["2d", "3d"]:
        raise KeyError("mode needs to be either 2d or 3d")

    pt_path = os.path.abspath(path)
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"No such file: {pt_path}")

    g = torch.load(pt_path, weights_only=False)

    # Expecting a torch_geometric.data.Data-like object
    if not hasattr(g, "pos") or not hasattr(g, "edge_index"):
        raise ValueError("Loaded object does not look like a torch_geometric Data with pos/edge_index")

    pos = g.pos
    edge_index = g.edge_index
    labels = infer_labels(g)
    origin_edges = getattr(g, "origin_edges", None)
    title = f"{os.path.basename(pt_path)}"

    # Print concise feature/shape info
    describe_graph_features(g)

    if mode == "3d":
        visualize_3d(pos, edge_index, labels, title, origin_edges)
    else:
        visualize_2d(pos, edge_index, labels, title, origin_edges)
        
    return

def main():
    parser = argparse.ArgumentParser(description="Visualize a PyG .pt graph (ligand/protein/merged)")
    parser.add_argument("--path", required=True, help="Path to a .pt file (single Data object)")
    parser.add_argument("--mode", choices=["2d", "3d"], default="3d", help="Visualization mode")
    args = parser.parse_args()

    visualize_pt_graph(args.path, args.mode)


if __name__ == "__main__":
    main()


