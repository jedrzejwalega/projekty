from Bio import SeqIO
from Bio.Seq import Seq
from Bio import Entrez
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import SearchIO


# Fetch the whole genome of Lotharella sp. plastid from GenBank:
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

# Running BLAST over the net:
# result_handle = NCBIWWW.qblast("blastn", "nt", protein_l_seq_fasta)
# with open("l_protein_blast.xml", "w") as out_handle:
#     out_handle.write(result_handle.read())
# result_handle.close()

result_handle = open("l_protein_blast.xml")

blast_records = NCBIXML.parse(result_handle)

# Information about BLAST results, maxi format:

# E_VALUE_THRESH = 0.04
# for blast_record in blast_records:
#     for alignment in blast_record.alignments:
#         for hsp in alignment.hsps:
#             if hsp.expect < E_VALUE_THRESH:
#                 print("****Alignment****")
#                 print("sequence:", alignment.title)
#                 print("length:", alignment.length)
#                 print("e value:", hsp.expect)
#                 print(hsp.query[0:75] + "...")
#                 print(hsp.match[0:75] + "...")
#                 print(hsp.sbjct[0:75] + "...")
#                 print("\n \n \n \n")


final_list = []
for blast_record in blast_records:
    for description in blast_record.descriptions:
        first_split = description.title.split("|")[4].strip(" ")
        second_split = first_split.split(" ")
        final_list.append(second_split[0])
        final_list.append(second_split[1])

string_list = []
for i in range(len(final_list)):
    if i % 2 == 0:
        final_name = final_list[i] + " " + final_list[i + 1]
        string_list.append(final_name)

print(string_list)

# Parsing BLAST using the parser:
blast_file = SearchIO.read("l_protein_blast.xml", "blast-xml")
print(blast_file)
print(blast_file.hit_keys)

# Finding out index of a particular hit:
print(blast_file.index("gi|553832027|gb|KF438023.1|"))

sorted_blast = blast_file.sort(in_place=False, reverse=True)
for hit in sorted_blast:
    print(hit.id, hit.seq_len)
# ^ Here we get longest hits first, so it's quite useful

# Info about a hit using SearchIO, not NCBIXML:
print(blast_file[3])
print("\n")
print(blast_file[3].seq_len)
print(blast_file[3].description)
print(blast_file[3].id)