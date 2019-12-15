from Bio import Entrez
from Bio import Seq
from Bio import SeqIO
from Bio.Data import CodonTable
import matplotlib as plt
from Bio.Blast import NCBIWWW
from Bio import SearchIO

def fetch_sequence():
    # Acquiring neccessary information about file to fetch and user's email (required for Entrez)
    seq_id = input("Please type your sequence's id: ")
    seq_db = input("Please enter the database to search (nucleotide or protein): ") # Can't be protein, cause blast won't work later!
    user_email = input("Please enter your email: ")

    # Retrieving the sequence in gb format:
    Entrez.email = user_email
    handle = Entrez.efetch(db=seq_db, id=seq_id, rettype="gb")
    file_sequence = SeqIO.read(handle, "gb")
    handle.close()
    return file_sequence

# no internet so instead of fetch_sequence I use sequence from memory:
sequence = SeqIO.read("sequence.gb", "gb")

# Returns a list of all cds locations in a file
def find_cds(file_sequence):
    cds = []
    for feature in file_sequence.features:
        if feature.type == "CDS":
            cds.append((feature.location, feature.qualifiers))
    return cds

# Transcribes all cds locations into RNA and aminoacid sequences
def print_cds(cds, file_sequence):
    tables = ",".join(sorted(list(CodonTable.ambiguous_dna_by_name)))
    table = input("From the available tables: \n \n" + tables + "\n \nplease choose table of interest: ")
    for sequence in cds:
        if sequence[0].strand == 1:
            print("\nGene name: ", sequence[1]["gene"], "\n")
            print("DNA sequence:\n\n", file_sequence.seq[sequence[0].start : sequence[0].end].transcribe())
            print("\n")
            print("Aminoacid sequence:\n\n", file_sequence.seq[sequence[0].start : sequence[0].end].translate(table=table, cds=True), "\n")
        if sequence[0].strand == -1:
            print(sequence[1]["gene"], "\n")
            print("DNA sequence:\n\n", file_sequence.seq[sequence[0].start : sequence[0].end].reverse_complement().transcribe())
            print("\n")
            print("Aminoacid sequence:\n\n", file_sequence.seq[sequence[0].start : sequence[0].end].reverse_complement().translate(table=table, cds=True), "\n")

# Turns retrieved gb sequence into fasta format
def into_fasta(sequence):
    fasta = sequence.format("fasta")
    return fasta

# Running BLAST over the Internet:
def blast(fasta):
    result_handle = NCBIWWW.qblast("blastn", "nt", fasta)
    with open("blast_result.xml", "w") as out_handle:
        blast_records = NCBIWWW.parse(out_handle)
    out_handle.close()
    for blast_record in blast_records:
        print(blast_record)

    
all_cds_locations = find_cds(sequence)

cds_rna_protein = print_cds(all_cds_locations, sequence)

fasta = into_fasta(sequence)

blast(fasta)