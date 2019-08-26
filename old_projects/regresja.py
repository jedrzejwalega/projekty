import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 



def x_and_y_data(seed1, seed2):
    np.random.seed(seed1)
    data_x = np.random.randint(low = 1, high = 100, size = 30)

    np.random.seed(seed2)
    data_y = np.random.randint(low = 1, high = 100, size = 30)
    return data_x, data_y

#FUNKCJE


def calculate_SST(data_y):
    mean = np.mean(data_y)
    SST = 0
    for y in data_y:
        difference = mean - y
        SST += difference **2
    return SST

def calculate_a(data_x, data_y):
    mean_x = np.mean(data_x)
    mean_y = np.mean(data_y)
    index = range(len(data_x))
    sum_x_times_y = 0
    sum_mean_minus_x_squared = 0
    
    for n in index:
        mean_minus_x = mean_x - data_x[n]
        sum_mean_minus_x_squared += mean_minus_x ** 2
        mean_minus_y = mean_y - data_y[n]
        x_times_y = mean_minus_x * mean_minus_y
        sum_x_times_y += x_times_y
    
    a = float(sum_x_times_y) / float(sum_mean_minus_x_squared)

    return a, mean_x, mean_y



def calculate_b(a, mean_x, mean_y):
    b = mean_y - a * mean_x
    return b


def calculate_regression(data_x, data_y):

    a, mean_x, mean_y = calculate_a(data_x, data_y)
    b = calculate_b(a, mean_x, mean_y)
    return a, b

def expected_y_values(a, b, data_x):
    expected_y_values = []
    for x in data_x:
        y = a * x + b
        expected_y_values.append(y)
    return expected_y_values

def calculate_SSE(expected_y, data_y):
    index = range(len(expected_y))
    SSE = 0
    for n in index:
        reduction = expected_y[n] - data_y[n]
        SSE += reduction ** 2
    return SSE

def calculate_SSR(SSE, SST):
    SSR = SST - SSE
    return SSR

def calculate_r_squared(SSR, SST):
    r_squared = float(SSR) / float(SST)
    return r_squared

def make_chart(data_x, data_y, expected_y):
    figure, axis = plt.subplots()
    axis.plot(data_x, expected_y)
    axis.scatter(data_x, data_y, marker = "D", color = "red")
    axis.set_title("Linear regression")
    axis.set_xlabel("x values")
    axis.set_ylabel("y values")

    plt.show()


def linear_regression(data_x, data_y):
    # data_x, data_y = x_and_y_data(seed1, seed2)
    SST = calculate_SST(data_y)
    a, b = calculate_regression(data_x, data_y)
    print("Regression: {a}x + {b}".format(a = a, b = b))
    expected_y = expected_y_values(a, b, data_x)
    SSE = calculate_SSE(expected_y, data_y)
    SSR = calculate_SSR (SSE, SST)
    r_squared = calculate_r_squared(SSR, SST)
    print("Coefficient of determination is equal to {coefficient}%.".format(coefficient = r_squared * 100))
    return expected_y
#MAIN

def main(seed1, seed2):
    data_x, data_y = x_and_y_data(seed1, seed2)

    expected_y = linear_regression(data_x, data_y)

    make_chart(data_x, data_y, expected_y)




print(main(42, 98))

