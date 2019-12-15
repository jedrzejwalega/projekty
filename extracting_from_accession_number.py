from Bio import Entrez, SeqIO


Entrez.email = "jedrzejwalega@gmail.com"

outgroups = open("/home/jedrzej/projekty/Dinoflagellates_accession_numbers", "r").read()
# print(outgroups)
outgroups_list = outgroups.split("\n")
# print(outgroups_list)

outgroups_by_species = [outgroups_list[n:n+6] for n in range(0, len(outgroups_list), 6)][:-1]
# for group in outgroups_by_species:
    # print(group)


# List of SSUrDNA accession numbers along with species name:
SSUrDNA = [(group[0], group[1]) for group in outgroups_by_species if group[1] != "x"]
# print(SSUrDNA)

# List of LSUrDNA accession numbers along with species name:
LSUrDNA = [(group[0], group[2]) for group in outgroups_by_species if group[2] != "x"]
# print(LSUrDNA)

# List of Hsp 90 (Hsp + SSU)
Hsp90 = [(group[0], group[3]) for group in outgroups_by_species if group[3] != "x"]
# print(Hsp90)

Hsp90_numbers = [entry[1] for entry in Hsp90]
LSUrDNA_numbers = [entry[1] for entry in LSUrDNA]
SSUrDNA_numbers = [entry[1] for entry in SSUrDNA]

print(Hsp90_numbers, "\n", len(Hsp90_numbers))

print(SSUrDNA_numbers, "\n")
print(LSUrDNA_numbers, "\n")
print(Hsp90_numbers)
Hsp90_numbers = ",".join(Hsp90_numbers)
LSUrDNA_numbers = ",".join(LSUrDNA_numbers)
SSUrDNA_numbers = ",".join(SSUrDNA_numbers)


bases = "assembly"


handle = Entrez.efetch(db=bases, id=Hsp90_numbers, rettype="gb")
file_handle = open("Hsp90_all", "w+")
file_handle.write(handle.read())
handle.close()
file_handle.close()

iterator = list(SeqIO.parse("/home/jedrzej/projekty/Hsp90_all", "genbank"))
for record in iterator:
    SeqIO.write(record, "/home/jedrzej/projekty/Dinoflagellates/Hsp90/" + record.annotations["organism"] + " Hsp + SSU", "gb")
    
