from Bio import SeqIO
from Bio import Entrez
import pandas as pd
import os
from urllib.error import HTTPError


Entrez.email = "jedrzejwalega@gmail.com"

table = pd.read_csv("/home/jedrzej/projekty/table_1.csv")

# for (column_name, column_data) in table.iteritems():
#     print("Column name: ", column_name[0])
#     # print("Column contents: ", column_data.values)

species_names = list(table["Binomial"])
acc_keys = []
for n in range(1, 74):
    for index, row in table.iterrows():
        acc_keys.append((row[3], row[4], str(row[4 + n]), table.columns[4 + n]))
# print(len(acc_keys))

acc_keys_2 = [key for key in acc_keys if key[0] != "www" if key[0] != "local" if key[2] != "nan"]
# print(acc_keys_2)
# print(len(acc_keys_2))

# os.mkdir("/media/jedrzej/8CC00C6CC00C5F38/tempDir")

# all_gene_names = [key[3] for key in acc_keys_2]
# print(all_gene_names)

# for gene_name in all_gene_names:
#         try:
#                 os.mkdir("/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/" + str(gene_name))
#         except FileExistsError:
#                 print("Directory already exists")


# Creating directories and subdirectories:

# for key in acc_keys_2:
#         try:
#                 os.mkdir("/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/" + str(key[3]))
#         except FileExistsError:
#                 print(" directory already present")
#         name = key[1]
#         name_split = name.split(" ")
#         try:
#                 os.mkdir("/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/" + str(key[3] + "/" + name_split[0]))
#         except FileExistsError:
#                 print("subdirectory already present")

# def fetching_data(lista):
#         databases = ["refseq", "db_est", "nr_nt"]

all_the_keys = [key[2] for key in acc_keys_2]
all_the_keys = ",".join(all_the_keys)
print(len(all_the_keys))

# handle = Entrez.efetch(db="nucleotide", id=all_the_keys, rettype="gb")
# parser = list(SeqIO.parse(handle, "gb"))
# print(len(parser))
# SeqIO.write(parser, "/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/big_nucleotide", "gb")
# fil = SeqIO.write(handle, "/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/big_nuccore", "gb")

# handle = Entrez.efetch(db="protein", id=all_the_keys, rettype="gb")
# fil = SeqIO.write(handle, "/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/big_protein", "gb")


# entries_list = []
# for key in acc_keys_2:
#         try:
#                 handle = Entrez.efetch(db="nucleotide", id=key[2], rettype="gb")
#                 output = SeqIO.read(handle, "gb")
#                 entries_list.append((output.id, key[1], key[3]))
#         except HTTPError:
#                 continue
#         try:
#                 new_dir = "/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/" + str(key[3])
#                 os.mkdir(new_dir)
#         except FileExistsError:
#                 print(" directory already present")
#         name = key[1]
#         name_split = name.split(" ")
#         try:
#                 new_dir = "/media/jedrzej/8CC00C6CC00C5F38/jedrzej/New_experiment/" + str(key[3]) + "/" + name_split[0]
#                 os.mkdir(new_dir)
#                 SeqIO.write(output, new_dir + "/" + key[1] + " " + key[3], "gb")
#         except FileExistsError:
#                 print("subdirectory already present")
#                 SeqIO.write(output, new_dir + "/" + key[1] + " " + key[3], "gb")
# print(len(entries_list))

handle = Entrez.efetch(db="protein", id=all_the_keys, rettype="gb")
parser = list(SeqIO.parse(handle, "gb"))
for parse in parser:
        print(parse.id)