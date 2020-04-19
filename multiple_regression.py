import numpy as np 
from sklearn.linear_model import LinearRegression

np.random.seed(23)
x1 = np.random.normal(24, 8, 100)
x2 = np.random.normal(43, 12, 100)
# print(x1, "\n", "\n", x2)
zipped = zip(x1, x2)
# print(list(zipped))

# Converting tuples into a list:
x = []
for x_1, x_2 in zipped:
    # print(x1, x2)
    x.append([x_1, x_2])
# print(x)

stutter = np.random.normal(5, 3, 100)
y = x1 * 2 + stutter
y.reshape(-1)

model = LinearRegression().fit(x, y)
r_squared = model.score(x, y)
print(r_squared)
b0 = model.intercept_
b1 = model.coef_
print(b0, "\n", b1)

pred_x1 = np.random.normal(20, 3, 10)
pred_x2 = np.random.normal(23, 3, 10)
pred_x = zip(pred_x1, pred_x2)
pred_x_lst = []
for x1, x2 in pred_x:
    # print(type(x1))
    pred_x_lst.append([x1, x2])

# print(pred_x_lst)
pred_y = model.predict(pred_x_lst)
print(pred_y)