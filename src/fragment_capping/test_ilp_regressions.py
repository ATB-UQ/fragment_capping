"""Regression tests for two ILP bugs (hand-built molecules, no data files needed).

1. ``get_all_tautomers`` deleted the sequential-solve constraints by the pre-2.x pulp names
   (``1_Sequence_Objective``) and raised KeyError on pulp >= 2.x (``Sequence_Objective_<i>``).
2. ``assign_bond_orders_and_charges_with_ILP(disallow_allenes_completely=True)`` raised
   ``Unknown variable type: A`` when reading the solution back.
"""
from fragment_capping.helpers.types_helpers import Atom
from fragment_capping.helpers.molecule import Molecule


def build(atoms, bonds, name):
    return Molecule(
        [Atom(index=i, element=e, valence=v, capped=True, coordinates=(0.0, 0.0, 0.0)) for (i, e, v) in atoms],
        bonds,
        name=name,
    )


def formamide():
    return build(
        [(1, 'C', 3), (2, 'O', 1), (3, 'N', 3), (4, 'H', 1), (5, 'H', 1), (6, 'H', 1)],
        [(1, 2), (1, 3), (1, 4), (3, 5), (3, 6)],
        'formamide',
    )


def hydrogens_per_heavy_atom(molecule, heavy_atom_ids):
    return tuple(
        sum(1 for bond in molecule.bonds if i in bond and any(j not in heavy_atom_ids for j in bond))
        for i in sorted(heavy_atom_ids)
    )


def test_get_all_tautomers_enumerates_past_the_first_solve():
    molecule = formamide()
    molecule.remove_all_hydrogens(mark_all_uncapped=True)
    tautomers = molecule.get_all_tautomers(total_number_hydrogens=3, net_charge=0)
    # H distributions over (C, O, N): formamide is (1, 0, 2), formimidic acid (1, 1, 1)
    distributions = {hydrogens_per_heavy_atom(t, {1, 2, 3}) for t in tautomers}
    assert distributions == {
        (1, 0, 2), (1, 1, 1), (1, 2, 0), (0, 0, 3), (0, 1, 2), (0, 2, 1), (0, 3, 0),
    }, distributions
    assert len(tautomers) == 7


def test_get_all_tautomers_naive_enumerates_past_the_first_solve():
    molecule = formamide()
    tautomers = molecule.get_all_tautomers_naive(total_number_hydrogens=3, net_charge=0, debug=None)
    assert len(tautomers) >= 2


def propene():
    return build(
        [(1, 'C', 3), (2, 'C', 3), (3, 'C', 4)] + [(i, 'H', 1) for i in range(4, 10)],
        [(1, 2), (2, 3), (1, 4), (1, 5), (2, 6), (3, 7), (3, 8), (3, 9)],
        'propene',
    )


def propadiene():
    return build(
        [(1, 'C', 3), (2, 'C', 2), (3, 'C', 3)] + [(i, 'H', 1) for i in range(4, 8)],
        [(1, 2), (2, 3), (1, 4), (1, 5), (3, 6), (3, 7)],
        'propadiene',
    )


def heavy_bond_orders(molecule):
    return {tuple(sorted(bond)): order for (bond, order) in molecule.bond_orders.items() if max(bond) <= 3}


def test_disallow_allenes_completely_without_allene():
    molecule = propene()
    molecule.assign_bond_orders_and_charges_with_ILP(disallow_allenes_completely=True)
    assert heavy_bond_orders(molecule) == {(1, 2): 2, (2, 3): 1}
    assert all(charge == 0 for charge in molecule.formal_charges.values())


def test_disallow_allenes_completely_with_allene():
    molecule = propadiene()
    molecule.assign_bond_orders_and_charges_with_ILP(disallow_allenes_completely=True)
    # =C= is forbidden: only one C=C survives, the middle/end carbons carry charges instead
    assert heavy_bond_orders(molecule) == {(1, 2): 2, (2, 3): 1}
    assert (molecule.formal_charges[2], molecule.formal_charges[3]) == (-1, -1)

    default = propadiene()
    default.assign_bond_orders_and_charges_with_ILP()
    assert heavy_bond_orders(default) == {(1, 2): 2, (2, 3): 2}
