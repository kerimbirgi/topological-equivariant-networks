from torch_geometric.data import Data
from etnn.combinatorial_data import Cell

# Base features: 4 bond type + conjugation + ring + length = 7
BASE_FEATURES = 7

def _bond_lift_core(graph: Data) -> set[Cell]:
    """Core bond lifting logic shared by both lifters."""
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

def bond_lift(graph: Data) -> set[Cell]:
    """Bond lifter for standard datasets (no cross-connections)."""
    return _bond_lift_core(graph)

def bond_lift_cross(graph: Data) -> set[Cell]:
    """Bond lifter for datasets with cross-connections."""
    return _bond_lift_core(graph)

# Set feature counts explicitly
bond_lift.num_features = BASE_FEATURES + 2        # 7 + 2 = 9 features (no cross)
bond_lift_cross.num_features = BASE_FEATURES + 3  # 7 + 3 = 10 features (with cross)