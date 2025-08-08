from torch_geometric.data import Data
from etnn.combinatorial_data import Cell

NUM_FEATURES = 7  # 4 bond type + conjugation + ring + length

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

bond_lift.num_features = NUM_FEATURES