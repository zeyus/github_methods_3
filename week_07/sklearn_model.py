# Create a function that squares the input
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def square(x):
    return x**2

# generate data, y, based on an input range of [0, 6] with a spacing of 0.1
x = np.arange(0, 6, 0.1)
x_squared = square(x)
#  add normally distributed noise to y with sigma=5 (set a seed to 7 `np.random.seed(7)`) to y and call it y_noisy
np.random.seed(7)
y_noisy = x + np.random.normal(0, 5, len(x))

# plot the true function and the generated points

plt.scatter(x, y_noisy, label='Generated points')
plt.plot(x, x_squared, label='True function')
plt.show()

# Fit a linear regression using `LinearRegression` from `sklearn.linear_model` based on y_noisy and x

regressor = LinearRegression()
fit = regressor.fit(x.reshape(-1, 1), y_noisy)
y_hat = fit.intercept_ + fit.coef_ * x

# plot the fitted line (see the `.intercept_` and `.coef_` attributes of the `regressor` object) on top of the plt
plt.scatter(x, y_noisy, label='Generated points')
plt.plot(x, x_squared, label='True function')
plt.plot(x, y_hat, label='Fitted line')
plt.show()



quadratic = PolynomialFeatures(degree=2)
X_quadratic = quadratic.fit_transform(x.reshape(-1, 1))
regressor_quadratic = LinearRegression()
fit_quadratic = regressor_quadratic.fit(X_quadratic, y_noisy)
y_quadratic_hat = fit_quadratic.predict(X_quadratic)
plt.scatter(x, y_noisy, label='Generated points')
plt.plot(x, x_squared, label='True function')
plt.plot(x, y_hat, label='Fitted line')
plt.plot(x, y_quadratic_hat, label='Fitted quadratic line')
plt.show()


p5 = PolynomialFeatures(degree=5)
X_p5 = p5.fit_transform(x.reshape(-1, 1))
regressor_p5 = LinearRegression()
fit_p5 = regressor_p5.fit(X_p5, y_noisy)
y_p5_hat = fit_p5.predict(X_p5)
plt.scatter(x, y_noisy, label='Generated points')
plt.plot(x, x_squared, label='True function')
plt.plot(x, y_hat, label='Fitted line')
plt.plot(x, y_quadratic_hat, label='Fitted quadratic line')
plt.plot(x, y_p5_hat, label='Fitted polynomial 5th degree')
plt.show()

