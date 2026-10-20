'''The Lewis-structure ILP rows, as pulp constraints.

``Molecule.assign_bond_orders_and_charges_with_ILP`` (molecule.py), ``get_all_tautomers`` /
``get_all_tautomers_naive`` (tautomers.py) and ``get_best_capped_molecule_with_ILP`` (capping.py)
state the same octet, absolute-charge, electron-balance and allene constraints.  The rows
themselves are built once, solver-agnostic, in ``chemistry_helpers.bond_order_model`` (D6:
one bond-order constraint model for the platform; the electron-bookkeeping encoding is this
package's).  This module makes the pulp variables and turns each shared row into a pulp
constraint under the name the LP has always carried.  Where the callers differ (which atoms and
bonds are in scope, constraint names) the difference is an explicit parameter: the LP each caller
writes is unchanged.  ``summation`` is accepted for compatibility and ignored: pulp writes the
terms of a constraint sorted by variable name, so ``sum`` and ``lpSum`` were never distinguishable.
'''
from typing import Any, Callable, Dict, Iterable, List, Set

from chemistry_helpers.bond_order_model import (
    INF, Expr, Row, absolute_charge_rows, allene_switch_rows, cumulene_row, electron_balance_row, octet_row)

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

def _expr(variable: Any) -> Expr:
    '''The shared model's expression for one pulp variable (keyed by its name).'''
    return Expr.var(variable.name)

def _add(problem: Any, row: Row, variables: Iterable[Any], name: Any = None) -> None:
    '''Add a shared :class:`Row` to the pulp problem over the pulp ``variables`` it names.'''
    from pulp import LpAffineExpression
    lookup = {variable.name: variable for variable in variables}
    expression = LpAffineExpression({lookup[variable_name]: coefficient for (variable_name, coefficient) in row.coefficients.items()})
    if row.is_equality:
        constraint = (expression == row.lower)
    elif row.upper != INF:
        constraint = (expression <= row.upper)
    else:
        constraint = (expression >= row.lower)
    if name is None:
        problem += constraint
    else:
        problem += constraint, name

def _atom_bond_variables(atom: Atom, bond_orders: Dict[Bond, Any], bonds: Iterable[Bond]) -> List[Any]:
    return [bond_orders[bond] for bond in bonds if atom.index in bond]

def add_electron_balance_constraint(
    problem: Any,
    charges: Dict[int, Any],
    atom: Atom,
    bond_orders: Dict[Bond, Any],
    bonds: Iterable[Bond],
    non_bonded_electrons: Dict[int, Any],
    electron_multiplier: int,
    name: Any,
    summation: Callable = sum,
    counter_charge: Any = None,
) -> None:
    '''``C_i == V_i - sum(bond orders) - multiplier * N_i`` under ``name`` (callers differ in the name).
    ``counter_charge``: the capping model's extra variable on a capping atom, subtracted from the right-hand side.'''
    atom_bonds = _atom_bond_variables(atom, bond_orders, bonds)
    charge = _expr(charges[atom.index]) + (_expr(counter_charge) if counter_charge is not None else 0)
    row = electron_balance_row(
        charge, VALENCE_ELECTRONS[atom.element], [_expr(bond) for bond in atom_bonds],
        _expr(non_bonded_electrons[atom.index]), electron_multiplier,
    )
    _add(problem, row, [charges[atom.index], non_bonded_electrons[atom.index]] + atom_bonds + ([counter_charge] if counter_charge is not None else []), name)

def add_absolute_charge_constraints(
    problem: Any,
    charges: Dict[int, Any],
    absolute_charges: Dict[int, Any],
    atom: Atom,
    name_suffixes: Any = ('1', '2'),
) -> None:
    '''|charge| <= absolute charge variable. ``name_suffixes`` are the two constraint-name tags (callers differ).'''
    rows = absolute_charge_rows(_expr(charges[atom.index]), _expr(absolute_charges[atom.index]))
    for (row, suffix) in zip(rows, name_suffixes):
        _add(problem, row, [charges[atom.index], absolute_charges[atom.index]], 'Absolute charge contraint {s} {i}'.format(s=suffix, i=atom.index))

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
        atom_bonds = _atom_bond_variables(atom, bond_orders, bonds)
        row = octet_row(
            [_expr(bond) for bond in atom_bonds], _expr(non_bonded_electrons[atom.index]), electron_multiplier,
            electrons=(2 if atom.element in {'H', 'HE'} else 8), electrons_per_bond=ELECTRONS_PER_BOND,
        )
        _add(problem, row, [non_bonded_electrons[atom.index]] + atom_bonds, 'Octet for atom {element}_{index}'.format(element=atom.element, index=atom.index))

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
        for row in allene_switch_rows(_expr(bond_orders[bond_1]), _expr(bond_orders[bond_2]), _expr(new_allene_switch)):
            _add(problem, row, [bond_orders[bond_1], bond_orders[bond_2], new_allene_switch])

def add_short_ring_allene_constraints(
    problem: Any,
    molecule: 'Molecule',
    bond_orders: Dict[Bond, Any],
    bonds: Iterable[Bond],
    disallow_allenes_in_small_rings: bool,
) -> None:
    '''No allene centres (C or N with two neighbours, summed bond order <= 3) in small rings. ``bonds`` are the bonds considered adjacent (callers differ).'''
    if not disallow_allenes_in_small_rings:
        return
    for atom in atoms_in_small_rings_for(molecule):
        if atom.element in {'C', 'N'}:
            adjacent_non_hydrogen_bonds = [bond for bond in bonds if atom.index in bond]
            if len(adjacent_non_hydrogen_bonds) == 2:
                atom_bonds = [bond_orders[bond] for bond in adjacent_non_hydrogen_bonds]
                # extra order = order - 1: at most one extra order <=> summed order <= 3
                row = cumulene_row(_expr(bond) - 1 for bond in atom_bonds)
                _add(problem, row, atom_bonds, 'No allenes for atom {atom_desc} in short ring'.format(atom_desc=atom_short_desc(atom)))
