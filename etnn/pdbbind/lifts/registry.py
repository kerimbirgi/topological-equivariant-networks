from .atom import atom_lift
from .bond import bond_lift, bond_lift_cross
from .bond_cross import bond_cross_lift
from .ring import ring_lift
from .supercell import supercell_lift

LIFTER_REGISTRY = {
    "atom": atom_lift,
    "bond": bond_lift,
    "bond_cross": bond_lift_cross,
    "bond_cross_lift": bond_cross_lift,
    "ring": ring_lift,
    "supercell": supercell_lift,
}