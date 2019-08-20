import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

np.random.seed(25)
data1 = np.random.normal(loc=0.0, scale=1.0, size=40)
data2 = np.random.uniform(low=0,high=1,size=40)
data3 = np.random.randint(low = 1, high = 100, size = 40)

figure, axis = plt.subplots()
axis.hist(data1)

figure2, axis2 = plt.subplots()
axis2.hist(data2)

figure3, axis3 = plt.subplots()
axis3.hist(data3)

plt.show()