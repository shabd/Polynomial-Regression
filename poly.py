import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#training set 
x_train = [[60], [80], [100], [140], [180]] # car speed mph
y_train = [[700], [900], [1300], [1750], [1800] #price in dollars 

# Testing set
x_test =[[60],[80], [110], [160]]
y_test = [[800], [1200], [1500], [1800]] #price in dollars

#train the model and plot a prediction 
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

 #Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Car price regressed on speed')
plt.xlabel('speed in mph')
plt.ylabel('Price in dollars')
plt.axis([0, 200, 0, 2000])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print (x_train)
print (X_train_quadratic)
print (x_test)
print (x_test_quadratic)