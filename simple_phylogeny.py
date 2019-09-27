from Bio import Phylo
import matplotlib.pyplot as plt
import copy


tree = Phylo.read("simple.dnd", "newick")
print(tree)
tree.branch_length = 0.27
print(tree.branch_length)
print(type(tree))
Phylo.draw(tree)

tree.branch_length = 0.82
print(tree.branch_length)
Phylo.draw(tree)

tree.rooted = True
# Phylo.draw(tree)

# Now to color the tree:
tree = tree.as_phyloxml()
print(type(tree))
tree.root.color = "blue"
Phylo.draw(tree)
mrca = tree.common_ancestor({"name" : "A"}, {"name" : "D"})
mrca.color = "salmon"
Phylo.draw(tree)

Phylo.write(tree, "tree_phyloxml", "phyloxml")

# Converting a tree to another format:
Phylo.convert("simple.dnd", "newick", "simple.xml", "nexml")

# Format function - doesn't create a new file 
tree1 = tree.format("newick")
print(tree1)

tree2 = tree.format("nexml")
print("\n \n \n \n \n")
print(tree2)

# Number of leafs (ends):
number = tree.count_terminals()
print(number)

# Length of branches:
lengths = tree.depths(unit_branch_lengths=True)
lengths2 = mrca.depths(unit_branch_lengths=True)
print(lengths)
print(lengths2)

# Removing nods:
new_tree = copy.deepcopy(tree)
new_tree.collapse("C")
Phylo.draw(new_tree)

# Ladderize (the most shallow branches go up, deeper ones go down):
new_tree = copy.deepcopy(tree)
new_tree.ladderize()
Phylo.draw(new_tree)

# Split:
new_tree = copy.deepcopy(tree)
mrca = new_tree.common_ancestor({"name" : "C"})
# mrca.split(n=4, branch_length = 0.23)
# Phylo.draw(new_tree)
# print(new_tree.get_terminals())
print(list(new_tree.find_clades(target="C")))

