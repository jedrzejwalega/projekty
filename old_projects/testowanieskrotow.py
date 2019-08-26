import numpy as np 
import pandas as pd 

series = pd.Series([0, 8, 24, 25, 89, np.nan])
print(series)

array = np.array([[9,8,12], [24, 89, 23], [90, 23, 11]])
print(array)

dataframe = pd.DataFrame(array, index = ["a", "b", "c"], columns = ["kolumna1", "kolumna2", "kolumna3"])
print(dataframe["kolumna1"])
