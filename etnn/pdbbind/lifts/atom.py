from torch_geometric.data import Data
from etnn.combinatorial_data import Cell

# In merged graphs we append a 1-d origin flag to the 13 base features → 14
NUM_FEATURES = 14

def atom_lift(graph: Data, **kwargs) -> set[Cell]:
    if (not hasattr(graph, "x")) or (graph.x is None):
        raise ValueError("The given graph does not have a feature matrix 'x'!")

    cells = set()
    X = graph.x
    for i in range(X.size(0)):
        # Map to ensure all features are floats
        cells.add((frozenset([i]), tuple(map(float, X[i].tolist()))))
    return cells

atom_lift.num_features = NUM_FEATURES
