
from Bio.Seq import Seq
from Bio.Alphabet.IUPAC import unambiguous_dna

sequence = Seq("""TTCGTAAACGAACCAGGTGATCAACGGTCTAGTCACAATGGCTCCTGCCCACGGGCCACAGCGCCCGAAC
TTCCTGCCGCGGCGGGGAGTCGTCAAGATGAGGATAGGCGACATGCGAGACGATGTAAACAAGAGGTGGA
AAAACCGGTTGATTCTTCTATCCGCCCTGTTCGTTGCCCTGTTCTTTGCTCCCTGGACCGCCACCCGTCG
GGAGGGGAGGAAAGCCCTAATCGCGAGAATACCGAGCAGCCTTCAAAGGAAAAGCATCATTGCTACAAAG
GGAGATTTTACGAGAAACATATTAGATATACGCTCCTCTTCACTCCAAAACATTGGGTTGATTCCTCGCC
GTCGTTATCATCATCATCAACAAGGTGGATGTAGCAACGAAAATGAAGGAGGAGGGGGAGGAGGAGACGG
AGGACGACATCCTCGGATATTCGCACAAAGCCAGAGTATGGACGATGACGATGATGATCAAGGTTTAGTA
GGGCGCAAATTAAAATTTGGATTGGCAGCACTGCTGTCGACGTGCCTCGGTGCCTCTTCCGCTTCTTCAG
GCGTTGAAGATGAAGGGCGGTCCACGGAGACTGGTGTCTCCCCCCCGACAACGAAGGATACTAATGGACT
ACGCAGCAATGAAAACAAAAACACTAGTAGCACCAATCCAAGTGGATGGAACATTACACAGTGGGACGAC
GTCTCAGAGATTATGTCGAATGTTACGCTCATATCCGATGTTGTAGAGCCAGAAGCTGTTATGAATATCG
CAGTCTTCACAGGTTTTCTCGGAACAAAGCCGACGGGAGAATCGCTTCAGCGACTTGTTGCGGGTATTCA
TGGTTACTACGCGAAGAAGGGTTTCTTATTCTGCCAGGTGAGCAGCCGAAGTATCATTCAGAACGGTTCA
GTGACCTTCAAGGCGACAGAACCTAGGACGAATGATCCTGAAGTAGTGATCAAATTCTTAAATCACAAGT
CCTGGGACACGATTCGGCGCGAAAATGAGGAAAAGAAACAACGGGTGAACAAGCAAAAGGAGAAGGAGGG
GAAAGAAGGAGGAGACAATATCGCAGCCTCAGCAGCATCAATGAAAAATGATGATAAGGGGGCTTCTTCT
TCTTCCTTGGATGAGAAAGGAGTGAAGGAGAAGAAGGAGGAGGAGGAGGATGGTGGAGTTATTGAGAAAC
CATCTGATCTCTTTGAAGTGACCGAGGGCAGAACTAAGCCAAGAGTCGTTCGAGATGCTTTGGGGTTGAA
GAAAGGGACGGTTACAAAATGGGATCCGGAAAAGTGGGCTGATCTCATCAACTCTGGGCTTTTCGAGGAG
GCTGTACCACAAATCACATGCAGTTCTGAAGGCAAAATCATGGTCATGTTGCTAGTAAGAGAACCAGAGC
GAAACGGCACTGGACTTGCCCCTGGCTGCACAATAGACCTTGGTTCCAGAAAGATAATGGGTGATATCAA
GTTCAAAGACCAGAACTTCTTAGGTCTCAATCAACACCTTAAGGCCCGTATATCTCGTCGAGGAAATTTC
TCATTTGAGGCAGACTGGGAAAATCCGATGCTTGGATCATCGTATAGTTATGCAGTAGGCGCTCGTATCC
AGAATTGGCGGGCACTAGGAGCTTTAATGAAAGGGGATCTCTTCTCATCCTCTTTCTCACCTCGAAAGAA
AACAAAAGAAAATGTCGGCGACAGCGAAGAGGGAGGAGCTAAATCATCCCTACAGGGTGCTCGAAGTAGC
GGAGAAGGGGGGGGGACAATCTTCGAAAACATTAAGTCAGAATCTTCTTCTTCTACTGCTGCTTCTTCAT
CATCATCAAAAACCCTCCCTGAGGGGAGGAATAATTTGGCTCTTGACCAAAATGGAAATCTGCTGATGAG
CTTAGGACAAGAAGATGAAAAGAACCCTGCTCACTACGATCAATTGCAGCTTCGATTACAAGTCATCAAT
TGTCATTTGAAACCTTGGATACTGTCATCAACGCTTGCAGGACTACACTCCACCACACCGATTGACCAGG
TGTCGTTTGAAGCACGGTTGGATGCTACTCACAAATCTCTGTTTGGAACGGACTCCTACGGCTTTGTCAC
ATCGCTACCACTATCGAACTCGCCACAGTACTGGAAGGCCTATGCGGATTTTGTCTCATCTGCGACGATA
AATTCAGAGGTCGCTGCGATGATGAATCTGCATTGTCATGTGGCGCCTCTTGTTCCTATCGATTCACAGG
TGCAATGGCTTGGCGGTCGTCAATCGGTGAGAGGCTATAAGGATGGAGAGATAGGTCCTGCATCATGTTG
GGTCAAGGGAACAGGCCAGGTATTCTATCCAGCCTCTGGGCAGGTGAAAGGCTTCTTTTTTGTAGATTCT
GTTCTTGGCAAAAAGATCCCCGGCAGAGACAGCTCACAGCAGCAGCAGTATTATCCCTATCAGCAGCATC
TCTCGTATCCTAGCGGTAGTAGCAGTACAGGTGGTAGTTTCGGACCTCTTTTCAACGACTCGCTCAAGGG
AGCTTCCGTAGGGCTTGGAGTGCAGGCAGGGCCATTATGTGCGGAGTTTGGAGTGACCAACAAGCTCACA
ACTAATTTGCATTTTACTATCGGTGGTGTCCCCGTGGATCTTAAGGTTGACAAATTGCTCTCGAGATTTT
CTCGAAATACAGGGGGGAGCAGTGGTCCTGGTAATAGAGGTGCCAGCAGTAGCAGCAGCACCACAGAGAT
GAAACCATCGGCAGCAGTATCAGAAGAACTGTCGGTCACATTCGAAGGGAAGAAAACTCAGGGTAGTACT
GCATATAAGTAGAGTCAAATGTGAGCAGGAGGCCTAGATAATGACTCCTAGGTCCATGCATGTGATGTGC
TATGTCATGGAGACTACCGCACATTATATTGATGATCACTCAACGTAGTTGCTTACAAAAAAAAAAAAA""", alphabet=unambiguous_dna)


print(len(sequence))
print(sequence.transcribe())