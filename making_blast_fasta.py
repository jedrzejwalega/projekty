from Bio import SeqIO
from Bio.Seq import Seq
from Bio import SearchIO
from Bio.Alphabet import generic_dna
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Alphabet import DNAAlphabet
from Bio import AlignIO
from Bio.Align.Applications import ClustalwCommandline
from Bio import Entrez



blast_file = SearchIO.read("l_protein_blast.xml", "blast-xml")
# print(blast_file, "\n")
# for hit in blast_file:
#     for hsp in hit:
        # print(hsp.query)

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
# print(to_fasta)

to_phylip = multiple_alignmenting(blast_file)
# print(to_phylip)
# SeqIO.write(to_fasta, "blast_fasta2.fasta", "fasta")
# AlignIO.write(to_phylip, "blast_phylip.phy", "phylip")

# file = AlignIO.convert("blast_phylip.phy", "phylip", "blast_fasta_final.fasta", "fasta")

# cline = ClustalwCommandline(infile = "blast_fasta_final.fasta")
# print(cline)
# stout, sterr = cline()


Entrez.email = "jedrzejwalega@gmail.com"
handle = Entrez.efetch(id="KF438023.1", rettype="gb", db="nucleotide")
genes = SeqIO.read(handle, "genbank")
handle.close()


def find_protein_l(genes):
    for feature in genes.features:
        if feature.type == "CDS":
            if feature.qualifiers["product"] == ["photosystem II protein L"]:
                my_gene = genes[feature.location.start : feature.location.end]
                if [my_gene.seq.translate(to_stop=True)] == feature.qualifiers["translation"]:
                    return my_gene
                else:
                    return "Error: translation of gene sequence is not equal to translation in feature."

l_protein_sequence = find_protein_l(genes)
print(l_protein_sequence.seq)

# SeqIO.write(l_protein_sequence, "l_fasta.fasta", "fasta")
