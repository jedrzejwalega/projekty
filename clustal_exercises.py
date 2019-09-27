from Bio.Align.Applications import ClustalwCommandline


cline = ClustalwCommandline(infile = "opuntia.fasta")
print(cline)
stout, sterr = cline()
