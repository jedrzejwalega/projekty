from Bio import SeqIO
from Bio import Entrez


Entrez.email = "jedrzejwalega@gmail.com"

# data = open("/home/jedrzej/projekty/accession_num", "r").read()
# data = data.split("\n")
# new_data = []
# for entry in data:
#     if "This study" not in entry and "unpublished data" not in entry:
#         new_data.append(entry)


# final_data = []
# for entry in new_data:
#     final_data.append(entry.split(" "))
# # print(final_data)

# acc_nums = []
# for entry in final_data:
#     acc_nums.append(entry[2:])
# # print(acc_nums)

# acc_nums_final = []
# for acc_num in acc_nums:
#     acc_num2 = ",".join(acc_num)
#     acc_nums_final.append(acc_num2)
# acc_nums_final = ",".join(acc_nums_final)
# # print(acc_nums)
# # print(len(acc_nums))

# handle = Entrez.efetch(db="homologene", id=acc_nums_final, rettype="gb")
# genbanks = SeqIO.parse(handle, "gb")
# for entry in genbanks:
#     name = entry.annotations["organism"] + " " + entry.id + " PsbO"
#     SeqIO.write(entry, "/home/jedrzej/projekty/Second- and third-hand chloroplasts in dinoflagellates: Phylogeny of oxygen-evolving enhancer 1 (PsbO) protein reveals replacement of a nuclear-encoded plastid gene by that of a haptophyte tertiary endosymbiont/PsbO/Used/" + name, "gb")

files = SeqIO.parse("/home/jedrzej/Downloads/sequence.gp", "gb")
for fil in files:
      name = fil.annotations["organism"] + " " + fil.id + "PsbO"
      SeqIO.write(fil, "/home/jedrzej/projekty/Second- and third-hand chloroplasts in dinoflagellates: Phylogeny of oxygen-evolving enhancer 1 (PsbO) protein reveals replacement of a nuclear-encoded plastid gene by that of a haptophyte tertiary endosymbiont/PsbO/NCBI/" + name, "gb")