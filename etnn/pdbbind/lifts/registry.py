from .atom import atom_lift
from .bond import bond_lift, bond_lift_cross
from .supercell import supercell_lift

LIFTER_REGISTRY = {
    "atom": atom_lift,
    "bond": bond_lift,
    "bond_cross": bond_lift_cross,
    "supercell": supercell_lift,
}