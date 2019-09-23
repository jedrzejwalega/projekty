from Bio import SeqIO
from Bio.Seq import Seq
from Bio import Entrez
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
import matplotlib.pyplot as plt


Entrez.email = "jedrzejwalega@gmail.com"
handle = Entrez.efetch(id="DQ851108.1", rettype="gb", db="nucleotide")
genes = SeqIO.read(handle, "genbank")
handle.close()


def all_cds_debugging(genes):
  cds = []
  for feature in genes.features:
    if feature.type == "CDS":
      cds.append(feature)
  return cds

def all_cds(genes):
  cds = []
  for feature in genes.features:
    if feature.type == "CDS":
      cds.append(feature.location)
  return cds

def translate_cds(cds, genes):
  counter = 0
  for sequence in cds:
    if sequence.strand == 1:
      counter += 1
      print(genes.seq[sequence.start: sequence.end].translate(table=11, to_stop=True, cds=True), "\n", counter)
    if sequence.strand == -1:
      counter += 1
      print(genes.seq[sequence.start: sequence.end].reverse_complement().translate(table=11, to_stop=True, cds=True), "\n", counter)

cds = all_cds(genes)
print(len(cds))

# translated_cds = translate_cds(cds, genes)
# print(cds)

def making_fastas(cds, genes):
    fastas = []
    
    for sequence in cds:
        if sequence.strand == 1:
            fast = genes.seq[sequence.start: sequence.end]
            
            fast2 = SeqRecord(seq=fast, id=genes.id, description=genes.description)
            fastas.append(fast2)
        if sequence.strand == -1:
            fast = genes.seq[sequence.start: sequence.end].reverse_complement()
            
            fast2 = SeqRecord(seq=fast, id=genes.id, description=genes.description)
            fastas.append(fast2)
    return fastas

to_fasta = making_fastas(cds, genes)

def multi_fasta(to_fasta):
    final = ""
    for fast in to_fasta:
        final += fast.format("fasta") + "\n"
    return final

print(multi_fasta(to_fasta))

all_fastas = multi_fasta(to_fasta)
#SUCCESS^


lengths = []
for sequence in cds:
  if sequence.strand == 1:
    fast = len(genes.seq[sequence.start: sequence.end])
    lengths.append(fast)

  if sequence.strand == -1:
    fast = len(genes.seq[sequence.start: sequence.end])
    lengths.append(fast)
  
print(lengths)

plt.hist(lengths, bins=20)
plt.show()