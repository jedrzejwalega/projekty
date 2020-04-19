from Bio import Phylo
tree = Phylo.read("opuntia.dnd", "newick") #file and type of tree
Phylo.draw_ascii(tree)