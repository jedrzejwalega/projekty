from Bio import SeqIO
from Bio.Seq import Seq
from Bio import SearchIO
from Bio.Alphabet import generic_dna
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Alphabet import DNAAlphabet
from Bio import AlignIO
from Bio.Align.Applications import ClustalwCommandline



blast_file = SearchIO.read("l_protein_blast.xml", "blast-xml")
# print(blast_file, "\n")
for hit in blast_file:
    for hsp in hit:
        print(hsp.query)

# def fasting(blast_file):
#     fasta_list = []
#     for hit in blast_file:
#         for hsp in hit:
#             fasta_list.append(SeqRecord(hsp.hit.seq, id=hit.id, description=hit.description))
#     return fasta_list

def fasting(blast_file):
    fasta_list = []
    fasta_list.append(blast_file[0][0].query)
    for hit in blast_file:
        for hsp in hit:
            fasta_list.append(hsp.hit)
    return fasta_list

def multiple_alignmenting(blast_file):
    multiple_alignment_list = []
    for hit in blast_file:
        for hsp in hit:
            multiple_alignment_list.append(MultipleSeqAlignment([hsp.query, hsp.hit]))   
    return multiple_alignment_list


to_fasta = fasting(blast_file)
print(to_fasta)

to_phylip = multiple_alignmenting(blast_file)
# print(to_phylip)
# SeqIO.write(to_fasta, "blast_fasta2.fasta", "fasta")
# AlignIO.write(to_phylip, "blast_phylip.phy", "phylip")

# file = AlignIO.convert("blast_phylip.phy", "phylip", "blast_fasta_final.fasta", "fasta")

cline = ClustalwCommandline(infile = "blast_fasta_final.fasta")
print(cline)
stout, sterr = cline()
