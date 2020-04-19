from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Alphabet import generic_dna
# alignment = AlignIO.read("PF05371_seed.sth", "stockholm")
# print(alignment)
# print(type(alignment))
# for record in alignment:
#     print("\n \n \n")
#     print(record)
#     print(type(record))
#     print(record.seq)
#     print(type(record.seq))
# for record in alignment:
#     print("\n \n")
#     print(record.dbxrefs)

alignments = AlignIO.parse("PF05371_seed.sth", "stockholm")
print(alignments)
print(type(alignments))
for record in alignments:
    print(record)
    print(type(record))

# ^ both alignment and alignments return the same thing, since there's only one entry in the multiple alignment

alignments = AlignIO.read("PF05371_seed.sth", "stockholm")
alignments_fasta = alignments.format("fasta")
print(alignments_fasta)
parsing = AlignIO.parse(AlignIO.read("PF05371_seed.sth", "stockholm"), "fasta", seq_count=2)
print(parsing)
for sth in parsing:
    print(sth)

# ^ wth isn't this working? the file is in FASTA format, so the parser should do its job and is supposedly doing (as evident by generator object parse in output), but somehow 
# the output from iteration isn't present


# Creating a MultipleAlignment file in phylip format:

m_alignment_1 = MultipleSeqAlignment([SeqRecord(seq=Seq("AATT"), id="Alpha"), SeqRecord(seq=Seq("GGCC"), id="Beta"), SeqRecord(seq=Seq("CCTT"), id="Gamma")])

# print(m_alignment_1)

m_alignment_2 = MultipleSeqAlignment([SeqRecord(seq=Seq("AAGT"), id="Alpha"), SeqRecord(seq=Seq("GGTC"), id="Beta"), SeqRecord(seq=Seq("CCGT"), id="Gamma")])

m_alignment_3 = MultipleSeqAlignment([SeqRecord(seq=Seq("AACT"), id="Alpha"), SeqRecord(seq=Seq("GGTT"), id="Beta"), SeqRecord(seq=Seq("CCAT"), id="Gamma")])

my_alignments = [m_alignment_1, m_alignment_2, m_alignment_3]

AlignIO.write(my_alignments, "testing_alignments.phy", "phylip")

# converting created phylip file to stockholm file:
AlignIO.convert("testing_alignments.phy", "phylip", "testing_alignments.sth", "stockholm")
# ^SUCCESS

# Creating a MultipleAlignment file with only one alignment in phylip and clustal format:
m_alignment_1 = MultipleSeqAlignment([SeqRecord(seq=Seq("AATT"), id="Alpha"), SeqRecord(seq=Seq("GGCC"), id="Beta"), SeqRecord(seq=Seq("CCTT"), id="Gamma")])
AlignIO.write(m_alignment_1, "single_alignment.phy", "phylip")
AlignIO.write(m_alignment_1, "single_alignment_clustal.aln", "clustal")

# read + write into another format:
file = AlignIO.read("PF05371_seed.sth", "stockholm")
AlignIO.write(file, "file.phy", "phylip")

# manipulating alignments:
print("Number of lines in MultipleAlignment file: ", len(file))
for line in file:
    print("Line sequence: ", line.seq)
    print("Line id: ", line.id)

print(file)
print(file[2:5])
print("second column: ", file[:, 1])
print("second column, third row: " ,file[2, 1])
print("rows 3-5, columns 1-6:", file[3:6, :6])

# removing fourth column:
edited_file = file[:, :4] + file[:, 5:]
print(edited_file)

# sorting rows by id:
print(file.sort())
print(type(file))
# this doesn't sort file for some reson, looks like it has to be sorted first and then printed o_O.


file.sort()
print(file)
print(type(file))
