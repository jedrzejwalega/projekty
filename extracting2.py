from Bio import SeqIO
from Bio import Entrez
from Bio import Seq

data_from_article = ["L" + str(n) for n in range(38627, 38643)]

data_from_article = ",".join(data_from_article)


Entrez.email = "jedrzejwalega@gmail.com"


# handle = Entrez.efetch("nuccore", id=data_from_article, rettype="gb")
# data = list(SeqIO.parse(handle, format="gb"))

# for entry in data:
#     name = entry.annotations["organism"] + " " + entry.id + " 24S rRNA"
#     SeqIO.write(entry, "/home/jedrzej/projekty/Dinoflagellates/Used/" + name, "gb")

acc_numbers = open("/home/jedrzej/projekty/acc_numbers", "r").read()
acc_numbers = acc_numbers.split("\n")
# acc_numbers = ",".join(acc_numbers)
# print(acc_numbers)

# handle = Entrez.efetch("nuccore", id=acc_numbers, rettype="gb")
# data = list(SeqIO.parse(handle, format="gb"))

# for entry in data:
#     name = entry.annotations["organism"] + " " + entry.id + " 24S rRNA"
#     SeqIO.write(entry, "/home/jedrzej/projekty/Dinoflagellates/From_NCBI/" + name, "gb")

# for num in acc_numbers:
#     if num not in data_from_article:
#         print(num)

entry_ids = []
handle = Entrez.efetch("nuccore", id=data_from_article, rettype="gb")
data = list(SeqIO.parse(handle, format="gb"))

unique_ids = []
for entry in data:
    entry_ids.append(entry.id)
for num in acc_numbers:
    if num not in entry_ids:
        unique_ids.append(num)
print(unique_ids)
unique_ids = ",".join(unique_ids)

handle = Entrez.efetch("nuccore", id=unique_ids, rettype="gb")
data = list(SeqIO.parse(handle, format="gb"))

for entry in data:
    name = entry.annotations["organism"] + " " + entry.id + " 24S rRNA"
    SeqIO.write(entry, "/home/jedrzej/projekty/Dinoflagellates/From_NCBI/" + name, "gb")