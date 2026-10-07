[![DOI](https://zenodo.org/badge/96262215.svg)](https://zenodo.org/badge/latestdoi/96262215)

# `fragment_capping`: An Integer Linear Program (ILP) for predicting Lewis structures and applications to capping molecular fragments and enumerating tautomers

## Introduction

This `python` module implements several graph-based Integer Linear Programs (ILPs)
which are described in the paper:

```
Caron, Bertrand; Engler, Martin; Mark, Alan E. and Klau, Gunnar W.
An Integer Linear Program (ILP) for predicting Lewis structures and applications to capping molecular fragments and enumerating tautomers
```

## Status and place in the platform

**Live-support.** Used on the request path by `website` (molecule images via
`helpers/molecule_img.py`, the `fragments` API's capping endpoint, `tasks/set_qm_bond_orders.py`) and
as the capping engine under `dihedral_fragments`, `dihedral_parameterization` and
`fragment_merger`. The platform's own bond-order assignment for standard InChI/SMILES is a
different, newer ILP (`atb_helpers.bond_orders`, scipy/HiGHS); this package is the paper's
original PuLP/CBC formulation and is no longer where new work goes.

## Install and dependencies

Editable install in the shared venv (`uv pip install --python /home/atb/ATB/.venv/bin/python -e
/home/atb/ATB/fragment_capping --no-deps`). Declared in `setup.cfg`: `pulp`, `networkx`, `numpy`;
`chemistry_helpers` (ATB sibling) and the `babel` executable are used at run time. There is no
`requirements.txt` (the old `make setup` target is gone). `graph_tool` and `chem_graph_tool`
are **optional** and imported lazily, only by the plotting/debug paths
(`helpers/molecule.py:290,614`); they are not installed in the shared venv.

Configuration (all optional): `ATB_FRAGMENT_CAPPING_SOLVER_MSG=true` (CBC output),
`ATB_FRAGMENT_CAPPING_DEBUG=true` / `ATB_FRAGMENT_CAPPING_DEBUG_DIR` (write failed ILPs out).
Solver time limits are in `config.py` (`ilp_solver`, `sequential_solve_within`; callers can bound
the whole bond-order ILP).

## Tests

`src/fragment_capping/test_*.py` are `unittest` classes that read relative paths
(`pdbs/*.pdb`), so run them from inside `src/fragment_capping/`. **They do not pass today**
(measured 2026-10-07): the `pdbs/` fixtures are not tracked (only a `.gitignore` is), and the
`test_capping_wang.py` cases build `Atom(coordinates=None)`, which `validated_atoms_dict` now
rejects (`helpers/molecule.py:1374`). The `Makefile` still targets an Anaconda 5.0.1 path.

## Algorithms and examples

### Lewis structure prediction
Given a complete molecular graph, returns the best Lewis structure of a molecule by distributing valence electrons to minimise the sum of absolute charge of the molecule and respecting relative atomic electronegativities.

```
from fragment_capping.helpers.molecule import Molecule, Atom

molecule = Molecule(
	atoms=[
		Atom(element='C', capped=True, index=1, valence=2, coordinates=(0, 0, 0)),
		Atom(element='O', capped=True, index=2, valence=1, coordinates=(1, 0, 0)),
		Atom(element='O', capped=True, index=3, valence=1, coordinates=(-1, 0, 0)),
	],
	bonds=[(1, 2), (1, 3)],
	name='CO2',
)

molecule.assign_bond_orders_and_charges_with_ILP()

print(molecule.mol2())
```

### Fragment capping
Given a partial molecule graph (**fragment**), find the smallest, most aliphatic, least charged molecule containing this fragment.

```
from fragment_capping.helpers.molecule import Molecule, Atom

fragment = Molecule(
	atoms=[
		Atom(element='C', capped=False, index=1, valence=None, coordinates=[0, 0, 0]),
		Atom(element='O', capped=True, index=2, valence=1, coordinates=(1, 0, 0)),
	],
	bonds=[(1, 2)],
	name='CO',
)

fragment.get_best_capped_molecule_with_ILP()

print(fragment.mol2())
```

### Tautomer enumeration
Tautomers are compounds that readily interconvert by the movement of an atom (usually hydrogen) or group of atoms from one site to another within the molecular structure (Katritzky **et al.**, 2010).
Given a molecule, enumerate all possible tautomers.

```
from fragment_capping.helpers.molecule import Molecule, Atom

molecule = Molecule(
	atoms=[
		Atom(element='C', capped=True, index=1, valence=3, coordinates=(0, 0, 0)),
		Atom(element='O', capped=True, index=2, valence=1, coordinates=(1, 0, 0)),
		Atom(element='H', capped=True, index=3, valence=1, coordinates=(0, 1, 0)),
		Atom(element='H', capped=True, index=4, valence=1, coordinates=(0, -1, 0)),
	],
	bonds=[(1, 2), (1, 3), (1, 4)],
	name='CH2O',
)

for tautomer in molecule.get_all_tautomers():
	print(tautomer.mol2())
```
