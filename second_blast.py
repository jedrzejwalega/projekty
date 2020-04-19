from Bio.Blast import NCBIXML
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
from Bio.Blast import NCBIWWW


Entrez.email = "jedrzejwalega@gmail.com"
handle = Entrez.efetch(id="KF438023.1", rettype="gb", db="nucleotide")
genes = SeqIO.read(handle, "genbank")
handle.close()

# From the genome select the DNA sequence of photosystem II protein L:
def find_protein_l(genes):
    for feature in genes.features:
        if feature.type == "CDS":
            if feature.qualifiers["product"] == ["photosystem II protein L"]:
                my_gene = genes[feature.location.start : feature.location.end]
                if [my_gene.seq.translate(to_stop=True)] == feature.qualifiers["translation"]:
                    return my_gene
                else:
                    return "Error: translation of gene sequence is not equal to translation in feature."

protein_l_seq = find_protein_l(genes)
protein_l_seq_fasta = protein_l_seq.format("fasta")
print(protein_l_seq_fasta)

# # Running BLAST over the net:
# result_handle = NCBIWWW.qblast("blastn", "nt", protein_l_seq_fasta)
# with open("l_protein_blast.xml", "w") as out_handle:
#     out_handle.write(result_handle.read())
# result_handle.close()

result_handle = open("l_protein_blast.xml")

blast_records = SearchIO.read(result_handle, "blast-xml")

# print(protein_l_seq_fasta)
# print(blast_records)

# SeqIO.write(protein_l_seq, "protein_l_seq.fasta", "fasta")


def fasting(blast_file):
    fasta_list = []

    for hit in blast_file:
        for hsp in hit:
            fasta_list.append(hsp.hit)
    return fasta_list

fasta_list = fasting(blast_records)
SeqIO.write(fasta_list, "protein_l_blast.fasta", "fasta")