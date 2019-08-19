import matplotlib.pyplot as plt
import pandas as pd 


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
print(iris.head())

#scatter
figure, axis = plt.subplots()
axis.scatter(iris[petal_length], iris[petal_width])
axis.set_title("Iris Dataset")
axis.set_xlabel("petal length")
axis.set_ylebel("petal width")
plt.show()