from .atom import atom_lift
from .bond import bond_lift

LIFTER_REGISTRY = {
    "atom": atom_lift,
    "bond": bond_lift,
}