from Bio import SeqIO
from Bio import Entrez
import pandas as pd
import numpy as np



table = pd.read_csv("table_1.csv")

for index, row in table.iterrows():
    print(row[3], row[4], row[5])
    print(row[3:6])