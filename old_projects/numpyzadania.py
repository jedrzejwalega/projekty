import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))

print(species_small)

unique_species = np.unique(species_small)
print(unique_species)

species_enumerated = np.ndenumerate(species_small)
print(list(species_enumerated))

for val in unique_species:
    print("\n")
    print(list(np.ndenumerate(species_small[species_small == val])))

co_chce = [index for index, species in np.ndenumerate(species_small[species_small == val]) for val in unique_species]
print(co_chce)
