#In the poly.py we find the relationship on how much a person earns based on their position Level
# and their salary

#Position              Level       Salary
#Business Analyst     |  1    |    45000
#Junior Consultant    |  2    |    50000
#Senior Consultant    |  3    |    60000
#Manager              |  4    |    80000
#Country Manager      |  5    |    110000
#Region Manager       |  6    |    150000
#Partner              |  7    |    200000
#Senior Partner       |  8    |    300000
#C-level              |  9    |    500000
#CEO                  | 10    |    1000000

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]] # Position Level
y = [[45000], [50000], [60000], [80000], [110000], [150000], [200000], [300000], [500000], [1000000]] #Salary

#Training the Linear Regression model on whole dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Training the Polynomial Regression model on whole dataset
poly_reg = PolynomialFeatures(degree= 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#the graph of the relationship between Salary of the person and their Position Level will look like
#this:

#plotting the data points
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green') #Predicting a new result with the Polynomial Regression
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()