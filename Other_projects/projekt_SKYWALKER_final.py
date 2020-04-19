import pandas as pd
from Bio import SeqIO
from Bio import Entrez
from Bio import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import ClustalwCommandline
from shutil import copyfile
import http


class gene():
    def __init__(self, start, stop, name, organism, sequence_id, seq):
        self.start = start
        self.stop = stop
        self.seq = seq
        self.name = name.strip("]").strip("[").strip("'").replace(" ", "_")
        self.organism = organism.strip("]").strip("[").strip("'").replace(" ", "_")
        self.sequence_id = sequence_id.replace(" ", "_")

    def make_fasta_title(self):
        fasta_title = self.organism + "\t" + self.sequence_id + "\t" + self.name
        self.fasta_title = fasta_title


class hsp():
    def __init__(self, sequence_id, start, stop):
        self.sequence_id = sequence_id
        self.start = start
        self.stop = stop

table = pd.read_csv("/home/jedrzej/Desktop/rpl26_blastn_alexandrium_tamarense_result3", sep="\t", names=["query_seq_id", "subject acc.ver", "percent_of_identical_matches", "alignment_length", "num_mismatches", "query_alignment_start", "query_alignment_end", "s. start", "s. end", "evalue", "bitscore", "unique_subj_scientific_names", "unique_subj_common_names", "unique_subj_blast_names", "unique_subj_kingdom_names", "subject_title"])
table2 = table[["subject acc.ver", "query_alignment_start", "query_alignment_end", "subject_title", "unique_subj_blast_names", "s. start", "s. end"]]
filtered_blast_table = table2[table2["unique_subj_blast_names"] == "dinoflagellates"]
filtered_blast_table["subject acc.ver"] = [x.split("|")[-2] for x in filtered_blast_table["subject acc.ver"]]


def main(path_to_blast_table, email, e_value_maximum_threshold=None, database="nuccore", desired_file_name="overlapping_genes"):
    # blast_table = read_blast_data_csv(path_to_blast_table)
    # filtered_blast_table = filter_e_value(blast_table, e_value_maximum_threshold)
    write_hit_params_to_files(filtered_blast_table, desired_file_name)
    sequences = get_sequences(email, database, desired_file_name)
    overlapping_genes = get_genes_with_hsp_overlap(sequences)
    overlapping_genes_with_titles = make_titles(overlapping_genes)
    title_dictionary = make_title_dictionary(overlapping_genes_with_titles)
    desired_file_name = desired_file_name + ".fasta"
    write_to_fasta(overlapping_genes_with_titles, desired_file_name)
    clustal(desired_file_name)
    desired_file_name_nexus = desired_file_name.replace("fasta", "nxs")
    desired_file_name_phylip = desired_file_name.replace("fasta", "ph")
    replace_tree_titles([desired_file_name, desired_file_name_nexus, desired_file_name_phylip], title_dictionary)
    write_dict_to_file(title_dictionary, desired_file_name)
    return title_dictionary



# read blast data table csv file
def read_blast_data_csv(path_to_blast_table):
    blast_table = pd.read_csv(path_to_blast_table, sep=",", names=["query acc.ver", "subject acc.ver", "% identity", "alignment length", "mismatches", "gap opens", "q. start", "q. end", "s. start", "s. end", "evalue", "bit score", "% positives", "query/sbjct frames"], index_col=False)
    return blast_table

# choose to filter table if e-value is given
def filter_e_value(blast_table, e_value_maximum_threshold):
    if e_value_maximum_threshold is None:
        return blast_table
    else:
        return filter_table_by_threshold(blast_table, e_value_maximum_threshold)


# filter the table by e-value
def filter_table_by_threshold(blast_table, e_value_maximum_threshold):
    filtered_blast_table = blast_table[blast_table["evalue"] <= e_value_maximum_threshold]
    return filtered_blast_table


# make files with hit parameters
def write_hit_params_to_files(filtered_blast_table, desired_file_name):
    write_only_acc_keys(filtered_blast_table, desired_file_name)
    write_all_three_params(filtered_blast_table, desired_file_name)


# create a new file "blast_results_only_acc_keys" with accession keys to blast results delimited by \n
def write_only_acc_keys(filtered_blast_table, desired_file_name):
    only_acc_keys = filtered_blast_table["subject acc.ver"]
    with open(desired_file_name + "_only_acc_keys", "w") as handle:
        for key in only_acc_keys:
            handle.write(str(key) + "\n")
    handle.close()


# create a new file "blast_results_all_three_params" with comma delimited accession keys, sequence start and sequence end. Each row is separated by \n
def write_all_three_params(filtered_blast_table, desired_file_name):
    all_three_params = filtered_blast_table[["subject acc.ver", "s. start", "s. end"]]

    with open(desired_file_name + "_all_three_params", "w") as handle:
        for index, row in all_three_params.iterrows():
            handle.write(str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "\n")
    handle.close()


# Save sequences from NCBI to new file
def get_sequences(email, database, desired_file_name):
    sequences = download_from_NCBI(email, database=database, desired_file_name=desired_file_name)
    write_sequences_to_file(sequences)
    return sequences


# Download sequences from NCBI
def download_from_NCBI(email, database, desired_file_name):
    Entrez.email = email
    accession_keys = ",".join(get_accession_keys(desired_file_name))
    print(accession_keys)
    handle = Entrez.efetch(db="nucleotide", id=accession_keys, rettype="gb")
    retrieved_sequences = list(SeqIO.parse(handle, "gb"))
    handle.close()
    
    return retrieved_sequences


# Retrieve a list of comma separated accessiom keys from "blast_results_only_acc_keys" file
def get_accession_keys(desired_file_name):
    with open(desired_file_name + "_only_acc_keys", "r") as handle:
        accession_keys = handle.read().split("\n")
    handle.close()
    return accession_keys[:-1]


# Write sequences to file
def write_sequences_to_file(sequences):
    SeqIO.write(sequences, "blast_hits", "gb")


# From all the found genes save only those which overlap, at least partially, with hsp regions
def get_genes_with_hsp_overlap(sequences):
    all_gene_locations = find_all_gene_locations(sequences)
    all_hsp = get_hsp_objects()
    overlapped_genes = filter_only_overlapping_genes(all_gene_locations, all_hsp)
    return overlapped_genes


# Find all cds or rRNA gene locations and return a list of them
def find_all_gene_locations(sequences):
    gene_locations = []
    for sequence in sequences:
        for feature in sequence.features:
            if (feature.type == "CDS") or (feature.type == "rRNA"):
                found_gene = make_gene(sequence, feature)
                gene_locations.append(found_gene)
    return gene_locations


# Make a gene object with its necessary information
def make_gene(sequence, feature):
    found_gene = gene(start=feature.location.start, stop=feature.location.end, name=get_name(feature), organism=get_organism(sequence), sequence_id=sequence.id, seq=feature.extract(sequence.seq))
    return found_gene


# Return gene's name from available info
def get_name(feature):
    if "gene" in feature.qualifiers.keys():
        return get_gene_name(feature)
    if "product" in feature.qualifiers.keys():
        return get_product_name(feature)
    else:
        return "Name unknown need of debugging"


# Return gene's name
def get_gene_name(feature):
    return str(feature.qualifiers["gene"])


# Return product's name
def get_product_name(feature):
    return str(feature.qualifiers["product"])


# Return source organism's name
def get_organism(sequence):
    for feature in sequence.features:
        if feature.type == "source":
            return str(feature.qualifiers["organism"])


# Return a list of hsp objects which contain information about their core parameters
def get_hsp_objects():
    hsp_objects = []
    hsp_parameters = read_hsp_file()
    for parameters in hsp_parameters:
        new_hsp = make_hsp(parameters)
        hsp_objects.append(new_hsp)
    return hsp_objects


# Read "blast_results_all_three_params" file and return a list of lists of parameters for single hsp
def read_hsp_file():
    hsp_parameters = []
    with open("dinos_all_three_params", "r") as handle:
        hsp_parameters_file_by_newline = handle.read().split("\n")
    handle.close()
    for three_parameters in hsp_parameters_file_by_newline[:-1]:
        parameters_for_single_hsp = three_parameters.split(",")
        hsp_parameters.append(parameters_for_single_hsp)
    return hsp_parameters


# Return a new hsp object
def make_hsp(parameters):
    new_hsp = hsp(sequence_id=parameters[0], start=parameters[1], stop=parameters[2])
    return new_hsp


# Return a set of genes which overlap with at least one hsp
def filter_only_overlapping_genes(all_gene_locations, all_hsp):
    all_matches = set()
    sequence_ids = set()
    for gene in all_gene_locations:
        for hsp in all_hsp:
            if (hsp.sequence_id == gene.sequence_id) and (gene.sequence_id not in sequence_ids):
                if (int(hsp.start) >= int(gene.start) and int(hsp.start) <= int(gene.stop)) or (int(hsp.stop) <= int(gene.stop) and int(hsp.stop) >= int(gene.start)):
                    all_matches.add(gene)
                    sequence_ids.add(gene.sequence_id)
    return all_matches

# Add fasta_title variable to the gene objects
def make_titles(overlapping_genes):
    for gene in overlapping_genes:
        gene.make_fasta_title()
    return overlapping_genes

# Return a dictionary with fasta titles from objects replaced with #n, where n is a number. Serves to give short sequence titles before using ClustalW
def make_title_dictionary(overlapping_genes_with_titles):
    n = 0
    title_dictionary = {}
    for gene in overlapping_genes_with_titles:
        gene.dictionary_key = "#" + str(n) + "_"
        n += 1
        title_dictionary[gene.dictionary_key] = gene.fasta_title
    return title_dictionary
    
# Write sequences to ,,overlapping_genes_files.fasta" file
def write_to_fasta(overlapping_genes_with_titles, desired_file_name):
    records = []
    for gene in overlapping_genes_with_titles:
        record = SeqRecord(seq=gene.seq, id=gene.dictionary_key, description="")
        records.append(record)
    SeqIO.write(records, desired_file_name, "fasta")

# Align sequences from fasta file in ClustalW
def clustal(desired_file_name):
    cline = ClustalwCommandline(align="yes", infile = desired_file_name, output="fasta")
    print(cline)
    stout, sterr = cline()
    
    cline = ClustalwCommandline(infile = desired_file_name, output="nexus")
    print(cline)
    stout, sterr = cline()

    cline = ClustalwCommandline(infile = desired_file_name, tree="yes")
    print(cline)
    stout, sterr = cline()


# Replace number_ tree titles with titles consisting of organism and gene/product name, separated by \t
def replace_tree_titles(tree_files, title_dictionary):
    
    for tree_file in tree_files:
        copy_files(tree_file)
        with open(tree_file, "r") as tree_to_replace:
            tree = tree_to_replace.read()

        for title in title_dictionary.keys():
            new_title = title_dictionary[title].replace("\t", "__")
            tree = tree.replace(title, new_title)
        
        with open(tree_file, "w") as tree_to_replace:
            tree_to_replace.write(tree)


# Make copies of files with names not replaced with values from dictionary
def copy_files(tree_file):
    copyfile(tree_file, tree_file.split(".")[0] + "_no_replacements." + tree_file.split(".")[1])


# Write name dictionary to separate file
def write_dict_to_file(title_dictionary, desired_file_name):
    with open(desired_file_name.replace(".fasta", "") + "_title_dictionary", "w") as handle:
        for key, value in title_dictionary.items():
            handle.write(key + ":" + value + "\n")

        
# MANUAL:
# Script generates .fasta, .ph and .nxs files, each one in two versions: raw (numerical) titles and titles replaced with those in dictionary. It also generates title dictionary.
# The script also creates two files - "your_desired_name_only_accession_keys" and "your_desired_name_all_three_params", with sequence accession keys and (sequence accession key, hit start, hit stop) information respectively. They come from filtered blast table after reducing its contents by the given e_value_maximum_threshold.


# Function call instructions (add function call at the last line of this file, since bash interface is not yet implemented):

# Invoke main() with parameters:
# path_to_blast_table = string with path to the csv table; REQUIRED
# email = string with email adress, required for Entrez to work; REQUIRED
# e_value_maximum_threshold = value to filter blast table with; OPTIONAL
# desired_file_name = string, will act as a prefix to the generated files; OPTIONAL, default = "overlapping_genes"
# database = string, name of the database to access the sequences by accesion number. Names od databases available at:
# https://www.ncbi.nlm.nih.gov/books/NBK25497/table/chapter2.T._entrez_unique_identifiers_ui/?report=objectonly

# Should the script freeze, it probably has to do with Entrez connection timeout. Aborting the script and starting it again should resolve the issue.

print(main("/home/jedrzej/Desktop/rpl26_blastn_alexandrium_tamarense_result3", "jedrzejwalega@gmail.com", desired_file_name="dinos", database="nt"))