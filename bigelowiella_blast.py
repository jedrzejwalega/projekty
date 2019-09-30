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
from Bio.Blast import NCBIXML
from Bio.Blast import NCBIWWW


Entrez.email = "jedrzejwalega@gmail.com"
handle = Entrez.efetch(id="DQ851108.1", rettype="gb", db="nucleotide")
bigelowiella_plastid = SeqIO.read(handle, "gb")
handle.close()

def find_psbk(bigelowiella_plastid):
    for feature in bigelowiella_plastid.features:
        if feature.type == "CDS":
            if feature.qualifiers["gene"] == ["psbK"]:
                psbk_sequence = bigelowiella_plastid[feature.location.start : feature.location.end]
                psbk_sequence.description = "Bigelowiella natans plastid photosystem II protein K"

    return psbk_sequence

psbk = find_psbk(bigelowiella_plastid)

# SeqIO.write(psbk, "psbk.fasta", "fasta")

psbk_fasta = psbk.format("fasta")


# Running BLAST over the net (FOR SOME REASON psbk_fasta works, but "psbk.fasta" doesn't, it has to be a variable):
# result_handle = NCBIWWW.qblast("blastn", "nt", psbk_fasta)
# with open("psbk_fasta.xml", "w") as out_handle:
#     out_handle.write(result_handle.read())
# result_handle.close()

result_handle = open("psbk_fasta.xml")

blast_records = SearchIO.read(result_handle, "blast-xml")

def blast_into_fasta(blast_records):
    hits_list = []
    for hit in blast_records:
        for hsp in hit:
            hits_list.append(hsp.hit)
    return hits_list

turning_blast_fasta = blast_into_fasta(blast_records)
print(turning_blast_fasta)
SeqIO.write(turning_blast_fasta, "psbk_blast.fasta", "fasta")