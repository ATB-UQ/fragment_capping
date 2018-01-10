from sys import stderr

from fragment_capping.helpers.types_helpers import Atom
from fragment_capping.helpers.molecule import Molecule, molecule_from_pdb_str

def example_porphyrin(use_ILP: bool = True) -> None:
    with open('pdbs/tetraphenylporphyrin.pdb') as fh:
        molecule = molecule_from_pdb_str(
            fh.read(),
            name='porphyrin',
        )
    molecule.remove_all_hydrogens()
    molecule.write_graph(
        'input',
        output_size=(1200, 1200),
        graph_kwargs={'include_atom_index': False},
    )
    tautomers = molecule.get_all_tautomers(
        net_charge=0,
        total_number_hydrogens=30,
        enforce_octet_rule=True,
        allow_radicals=False,
    )
    for (n, molecule) in enumerate(tautomers):
        molecule.write_graph(
            'tautomer_{n}'.format(n=n),
            output_size=(1200, 1200),
            graph_kwargs={'include_atom_index': False},
        )

def example_methylimidazole(use_ILP: bool = True) -> None:
    with open('pdbs/methylimidazole.pdb') as fh:
        molecule = molecule_from_pdb_str(
            fh.read(),
            name='methylimidazole',
        )
    molecule.remove_all_hydrogens()
    molecule.write_graph(
        'input',
        output_size=(600, 600),
        graph_kwargs={'include_atom_index': False},
    )
    tautomers = molecule.get_all_tautomers(
        net_charge=0,
        total_number_hydrogens=6,
        enforce_octet_rule=True,
        allow_radicals=False,
    )
    for (n, molecule) in enumerate(tautomers):
        molecule.write_graph(
            'tautomer_{n}'.format(n=n),
            output_size=(600, 600),
            graph_kwargs={'include_atom_index': False},
        )

def example_benzene():
    with open('pdbs/benzene.pdb') as fh:
        molecule = molecule_from_pdb_str(
            fh.read(),
            name='benzene',
        )
    molecule.remove_all_hydrogens()
    molecule.write_graph(
        'input',
        output_size=(600, 600),
        graph_kwargs={'include_atom_index': False},
    )
    tautomers = molecule.get_all_tautomers(
        net_charge=0,
        total_number_hydrogens=6,
        enforce_octet_rule=True,
        allow_radicals=False,
    )
    for (n, molecule) in enumerate(tautomers):
        molecule.write_graph(
            'tautomer_{n}'.format(n=n),
            output_size=(600, 600),
            graph_kwargs={'include_atom_index': False},
        )

def example_ethanal():
    with open('pdbs/ethanal.pdb') as fh:
        molecule = molecule_from_pdb_str(
            fh.read(),
            name='ethanal',
        )
    molecule.remove_all_hydrogens()
    molecule.write_graph(
        'input',
        output_size=(600, 600),
        graph_kwargs={'include_atom_index': False},
    )
    tautomers = molecule.get_all_tautomers(
        net_charge=0,
        total_number_hydrogens=4,
        enforce_octet_rule=True,
        allow_radicals=False,
    )
    for (n, molecule) in enumerate(tautomers):
        molecule.write_graph(
            'tautomer_{n}'.format(n=n),
            output_size=(600, 600),
            graph_kwargs={'include_atom_index': True},
        )

ALL_EXAMPLES = [
    #example_benzene,
    example_ethanal,
    #example_methylimidazole,
    #example_porphyrin,
]

def main() -> None:
    for example in ALL_EXAMPLES:
        try:
            example()
        except AssertionError as e:
            print(str(e))
            raise

if __name__ == '__main__':
    main()