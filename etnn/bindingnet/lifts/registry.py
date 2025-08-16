from .atom import atom_lift
from .bond import bond_lift
from .supercell import supercell_lift

LIFTER_REGISTRY = {
    "atom": atom_lift,
    "bond": bond_lift,
    "supercell": supercell_lift,
}