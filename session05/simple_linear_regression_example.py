import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.metrics import *
import pandas as pd
import os
from sklearn.datasets import fetch_mldata

xs = np.linspace(0.5, 20.5, 200)
ys = 5*(xs+2*np.random.randn(xs.shape[0])) + 500

# reshaping input to 2d arrays because the code expects that form. It will print an error telling you how to do this if you give it 1d arrays.
xs = xs.reshape(xs.shape[0], -1)
ys = ys.reshape(ys.shape[0], -1)

fraction_of_data_to_save_for_testing = 0.5
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xs, ys, test_size=fraction_of_data_to_save_for_testing)

#creating your machine learning model object
regr = linear_model.LinearRegression()

#fitting your machine learning model object
regr.fit(xtrain, ytrain)

#predicting values using your fitted model
ypred = regr.predict(xtest)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(ytest, ypred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(ytest, ypred))

plt.figure()
plt.scatter(xtest[:,0], ytest,  color='red', label="Testing Data")
plt.scatter(xtrain[:,0], ytrain, facecolors="none", edgecolors="black", label="Training Data")
plt.plot(xtest[:,0], ypred, color='blue', linewidth=3, label="Prediction")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()
