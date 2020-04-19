from Bio import SeqIO
from Bio.Seq import Seq
from Bio import Entrez
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.Alphabet import IUPAC


genes = SeqIO.read("sequence.fasta", "fasta")

# print(genes.id)
# print(genes.description)
# print(genes.seq)
# print(len(genes))

genes2 = SeqIO.read("sequence.gb", "gb")
print(genes2.id)
record = SeqRecord(seq = genes2.seq, id = genes2.id, description = genes2.description)
print(genes2.format("fasta"))
print(record.format("fasta"))
print(record.seq.alphabet)
print(type(genes2))