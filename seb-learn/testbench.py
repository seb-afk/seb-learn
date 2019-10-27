import numpy as np
import os
from sklearn.linear_model import LinearRegression
from data.data_generators import gen_sinusoidal
from data.data_processing import create_design_matrix
import linear_regression

x_mat, y = gen_sinusoidal(10, seed=12)
print(y.flatten())
x_design = create_design_matrix(2,x_mat)
print(x_design.flatten())
model = LinearRegression()
model.fit(x_design, y)
print(model.intercept_)
print(model.coef_)

print("My model")
model2 = linear_regression.LinearRegression()
model2.fit(x_design, y)

print(model2.weights_)
