from torch_geometric.data import Data
from etnn.combinatorial_data import Cell

# Base features: 7 bond type + conjugation + ring + length = 10
M_RBF = 16
K_INTER = 5
BASE_FEATURES = 9 + M_RBF

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

def bond_lift(graph: Data, **kwargs) -> set[Cell]:
    """Bond lifter for standard datasets (no cross-connections)."""
    return _bond_lift_core(graph)

def bond_lift_cross(graph: Data, **kwargs) -> set[Cell]:
    """Bond lifter for datasets with cross-connections."""
    return _bond_lift_core(graph)

# Set feature counts explicitly
bond_lift.num_features       = BASE_FEATURES + K_INTER + 2  # no cross: +2 type one-hot
bond_lift_cross.num_features = BASE_FEATURES + K_INTER + 3  # with cross: +3 type one-hot