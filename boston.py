import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sklearn
import sklearn.datasets


boston_dataset = sklearn.datasets.load_boston()
print(boston_dataset.keys())

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(boston)

boston["MEDV"] = boston_dataset.target
print(boston)

print(boston.isnull().sum())

# boston.plot.hist(subplots=True)
# plt.show()

figure, axis = plt.subplots()
axis.hist(boston["MEDV"])
plt.show()

correlation = boston.corr()
print(correlation)