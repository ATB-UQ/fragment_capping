fragments: protein_fragments.pickle
	python molecule_for_fragment.py
.PHONY: fragments

protein_fragments.pickle:
	scp scmb-atb.biosci.uq.edu.au:/home/uqbcaron/ATB/ivan_dihedrals/$@ .
