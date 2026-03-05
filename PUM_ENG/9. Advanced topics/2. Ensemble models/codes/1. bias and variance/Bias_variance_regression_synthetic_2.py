import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import pi as PI
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# for difference between np.polyfit ans sklearn PolynomialFreatures see:
# https://towardsdatascience.com/polynomial-regression-with-scikit-learn-what-you-should-know-bed9d3296f2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


from mlxtend.evaluate import bias_variance_decomp

plt.style.use('ggplot')
# Generate data
# number of observations
NUM_OBS = 500

# generate 500 random observations
x = np.linspace(3, 8, NUM_OBS)
y = np.sin(x) + np.random.normal(0, 0.2, NUM_OBS)

# plot x vs y
#plt.plot(x, y, 'o')

# Modelling
# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# plot training and testing sets
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'o')

# fit polynomial regression model
# degree of polynomial
poly_deg = 3

# fit model with sklearn
model = make_pipeline(PolynomialFeatures(poly_deg),LinearRegression())
model.fit(x_train.reshape(-1, 1), y_train)

plt.plot(x, y, 'o')
plt.plot(x, model.predict(x.reshape(-1, 1)), '-')

y_train_pred = model.predict(x_train.reshape(-1, 1))
y_test_pred = model.predict(x_test.reshape(-1, 1))

# compute train and test MSE  (mean squared error)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(train_mse, test_mse)


# perform bias-variance decomposition
mse, bias, variance = bias_variance_decomp(model, x_train.reshape(-1, 1), y_train, x_test.reshape(-1, 1), y_test, loss='mse', num_rounds=100, random_seed=1)
print(mse, bias, variance)
print(poly_deg, bias/variance)

num_models = 20
mse_list = []
bias_list = []
variance_list = []
bias_variance_list = []

for k in range(0, num_models):
    # fit model with sklearn
    model = make_pipeline(PolynomialFeatures(k+1),LinearRegression())
    model.fit(x_train.reshape(-1, 1), y_train)

    y_train_pred = model.predict(x_train.reshape(-1, 1))
    y_test_pred = model.predict(x_test.reshape(-1, 1))

    # perform bias-variance decomposition
    mse, bias, variance = bias_variance_decomp(model, x_train.reshape(-1, 1), y_train, x_test.reshape(-1, 1), y_test, loss='mse', num_rounds=100, random_seed=1)

    mse_list.append(mse)
    bias_list.append(bias)
    variance_list.append(variance)
    bias_variance_list.append(bias/variance)


# plot variance vs bias


# plot bias-variance trade-off
plt.plot(range(0, num_models), bias_variance_list, 'o')
plt.xlabel('Polynomial degree')
plt.ylabel('Bias-variance trade-off')


plt.plot(range(0, num_models), mse_list, 'o')
plt.xlabel('Polynomial degree')
plt.ylabel('MSE')

plt.plot(range(0, num_models), bias_list, 'o')
plt.xlabel('Polynomial degree')
plt.ylabel('bias')

plt.plot(range(0, num_models), variance_list, 'o')
plt.xlabel('Polynomial degree')
plt.ylabel('Variance')





