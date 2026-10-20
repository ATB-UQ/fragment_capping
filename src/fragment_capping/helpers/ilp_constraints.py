'''Constraint builders shared by the Lewis-structure ILPs.

``Molecule.assign_bond_orders_and_charges_with_ILP`` (molecule.py), ``get_all_tautomers`` /
``get_all_tautomers_naive`` (tautomers.py) and ``get_best_capped_molecule_with_ILP`` (capping.py)
all state the same octet, absolute-charge, electron-balance and allene constraints. They live here
once. Where the callers differ (which atoms and bonds are in scope, constraint names, ``sum`` vs
``lpSum``) the difference is an explicit parameter: the LP each caller writes is unchanged.
'''
from typing import Any, Callable, Dict, Iterable, List, Set

from fragment_capping.helpers.types_helpers import Atom, Bond
from fragment_capping.helpers.parameters import MAX_ABSOLUTE_CHARGE, MIN_ABSOLUTE_CHARGE, MAX_NONBONDED_ELECTRONS, VALENCE_ELECTRONS, ELECTRONS_PER_BOND
from fragment_capping.helpers.misc import atom_short_desc
from fragment_capping.helpers.rings import atoms_in_small_rings_for

OCTET_EXEMPT_ELEMENTS = {'B', 'BE', 'P', 'S'}

def make_charge_variables(atom_ids: Iterable[int]) -> Dict[int, Any]:
    from pulp import LpVariable, LpInteger
    return {
        atom_id: LpVariable("C_{i}".format(i=atom_id), -MAX_ABSOLUTE_CHARGE, MAX_ABSOLUTE_CHARGE, LpInteger)
        for atom_id in atom_ids
    }

def make_absolute_charge_variables(atom_ids: Iterable[int]) -> Dict[int, Any]:
    '''Extra variables used to bind the charges (|C_i| <= Z_i, see ``add_absolute_charge_constraints``).'''
    from pulp import LpVariable, LpInteger
    return {
        atom_id: LpVariable("Z_{i}".format(i=atom_id), MIN_ABSOLUTE_CHARGE, MAX_ABSOLUTE_CHARGE, LpInteger)
        for atom_id in atom_ids
    }

def make_non_bonded_electron_variables(atom_ids: Iterable[int], electron_multiplier: int) -> Dict[int, Any]:
    from pulp import LpVariable, LpInteger
    return {
        atom_id: LpVariable("N_{i}".format(i=atom_id), 0, MAX_NONBONDED_ELECTRONS // electron_multiplier, LpInteger)
        for atom_id in atom_ids
    }

def electron_balance_rhs(
    atom: Atom,
    bond_orders: Dict[Bond, Any],
    bonds: Iterable[Bond],
    non_bonded_electrons: Dict[int, Any],
    electron_multiplier: int,
    summation: Callable = sum,
) -> Any:
    '''Right-hand side of the electron balance: formal charge = valence electrons - bond orders - non-bonded electrons.'''
    return (
        VALENCE_ELECTRONS[atom.element]
        - summation([bond_orders[bond] for bond in bonds if atom.index in bond])
        - electron_multiplier * non_bonded_electrons[atom.index]
    )

def add_absolute_charge_constraints(
    problem: Any,
    charges: Dict[int, Any],
    absolute_charges: Dict[int, Any],
    atom: Atom,
    name_suffixes: Any = ('1', '2'),
) -> None:
    '''|charge| <= absolute charge variable. ``name_suffixes`` are the two constraint-name tags (callers differ).'''
    problem += charges[atom.index] <= absolute_charges[atom.index], 'Absolute charge contraint {s} {i}'.format(s=name_suffixes[0], i=atom.index)
    problem += -charges[atom.index] <= absolute_charges[atom.index], 'Absolute charge contraint {s} {i}'.format(s=name_suffixes[1], i=atom.index)

def add_octet_constraint(
    problem: Any,
    atom: Atom,
    bond_orders: Dict[Bond, Any],
    bonds: Iterable[Bond],
    non_bonded_electrons: Dict[int, Any],
    electron_multiplier: int,
    summation: Callable = sum,
) -> None:
    '''Octet (duet for H, HE) for ``atom``; B, BE, P and S are exempt. The caller decides whether the rule is enforced and for which atoms.'''
    if atom.element not in OCTET_EXEMPT_ELEMENTS:
        problem += (
            ELECTRONS_PER_BOND * summation([bond_orders[bond] for bond in bonds if atom.index in bond]) + electron_multiplier * non_bonded_electrons[atom.index] == (2 if atom.element in {'H', 'HE'} else 8),
            'Octet for atom {element}_{index}'.format(element=atom.element, index=atom.index),
        )

def non_allene_atoms_for(atoms: Iterable[Atom], bonds: Iterable[Bond], disallow_allenes_completely: bool) -> Dict[Atom, List[Bond]]:
    '''Two-neighbour carbons (candidate allene centres) mapped to their two bonds; empty unless allenes are disallowed.'''
    non_allene_atoms = {}
    for atom in atoms:
        if disallow_allenes_completely:
            atom_bonds = [bond for bond in bonds if atom.index in bond]
            if atom.element == 'C' and len(atom_bonds) == 2:
                non_allene_atoms[atom] = atom_bonds
    return non_allene_atoms

def add_allene_exclusion_constraints(problem: Any, bond_orders: Dict[Bond, Any], non_allene_atoms: Dict[Atom, List[Bond]]) -> None:
    '''Forbid =C= : the two bond orders of a two-neighbour carbon may not both be 2 (switch variable A_i).'''
    from pulp import LpVariable, LpBinary
    for (atom, (bond_1, bond_2)) in non_allene_atoms.items():
        new_allene_switch = LpVariable('A_{i}'.format(i=atom.index), 0, 1, LpBinary)
        problem += 2 * bond_orders[bond_1] - bond_orders[bond_2] + 4 * new_allene_switch >= 3
        problem += 2 * bond_orders[bond_1] - bond_orders[bond_2] + 4 * new_allene_switch <= 5

def add_short_ring_allene_constraints(
    problem: Any,
    molecule: 'Molecule',
    bond_orders: Dict[Bond, Any],
    bonds: Iterable[Bond],
    disallow_allenes_in_small_rings: bool,
) -> None:
    '''No allene centres (C or N with two neighbours, summed bond order <= 3) in small rings. ``bonds`` are the bonds considered adjacent (callers differ).'''
    for atom in atoms_in_small_rings_for(molecule):
        if disallow_allenes_in_small_rings:
            if atom.element in {'C', 'N'}:
                adjacent_non_hydrogen_bonds = [bond for bond in bonds if atom.index in bond]
                if len(adjacent_non_hydrogen_bonds) == 2:
                    problem += sum(bond_orders[bond] for bond in adjacent_non_hydrogen_bonds) <= 3, 'No allenes for atom {atom_desc} in short ring'.format(atom_desc=atom_short_desc(atom))
