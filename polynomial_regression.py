import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(84)
data_x = np.random.normal(82, 40, 100)

data_x = np.array(data_x).reshape(-1, 1)

stutter = np.random.normal(5, 3, 100).reshape(-1, 1)
data_y = data_x ** 2 + stutter
print(data_x, "\n", "\n")
print(data_y)
print(len(data_y))
data_y = data_y.reshape(-1)


x_ = PolynomialFeatures(degree=2, include_bias= False)
x_ = x_.fit(data_x)
x_2 = x_.transform(data_x)
print(data_x)
print("\n", x_)
print("\n", x_2)

model = LinearRegression().fit(x_2, data_y)
r_squared = model.score(x_2, data_y)
print(r_squared)
bn = model.coef_
b0 = model.intercept_
print(b0, bn)