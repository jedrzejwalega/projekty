import numpy as np
from sklearn.linear_model import LinearRegression


np.random.seed(84)
data_x = np.random.normal(82, 40, 100)

data_x = np.array(data_x).reshape(-1, 1)

data_y = data_x ** 2 + 18
data_y = data_y.reshape(-1)

model = LinearRegression().fit(data_x, data_y)
r_squared = model.score(data_x, data_y)
print(r_squared)
b0 = model.intercept_
b1 = model.coef_
print(b0)
print(b1)

additional_x = np.array([139, 152, 160]).reshape(-1, 1)
y_predicted = model.predict(additional_x)
print(y_predicted)