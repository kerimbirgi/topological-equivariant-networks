from torch_geometric.data import Data
from etnn.combinatorial_data import Cell

# Base features: 4 bond type + conjugation + ring + length = 7
# Edge type features: +2 (no cross) or +3 (with cross) 
BASE_FEATURES = 7

def bond_lift(graph: Data) -> set[Cell]:
    cells = set()
    seen = set()
    ei = graph.edge_index.t().tolist()
    EA = graph.edge_attr
    for k, (u, v) in enumerate(ei):
        if u == v:
            continue
        key = frozenset([u, v])
        if key in seen:
            continue
        seen.add(key)
        cells.add((key, tuple(map(float, EA[k].tolist()))))
    return cells

def get_bond_num_features(graph: Data) -> int:
    """Determine number of features based on edge_attr dimensions"""
    if graph.edge_attr is not None:
        return graph.edge_attr.size(1)
    return BASE_FEATURES

# Set a default, but it will be dynamically determined
bond_lift.num_features = BASE_FEATURES + 2  # Default for no cross-connection