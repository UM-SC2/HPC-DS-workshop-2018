import matplotlib.pyplot as plt
import numpy as np
import sklearn
#from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import pandas as pd
import os
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import RobustScaler


housing = pd.read_csv("./housing.data", delim_whitespace=True, header=-1)
xhous = housing.iloc[:,:-1]
xs = np.array(xhous)
### scaling is critical for the success of this algorithm with this dataset!
xs = RobustScaler(quantile_range=(25, 75)).fit_transform(xs)

yhous = housing.iloc[:,-1]
ys = np.array(yhous)

# reshaping input to 2d arrays because the code expects that form. It will print an error telling you how to do this if you give it 1d arrays.
#xs = xs.reshape(xs.shape[0], -1)
#ys = ys.reshape(ys.shape[0], -1)

fraction_of_data_to_save_for_testing = 0.5
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xs, ys, test_size=fraction_of_data_to_save_for_testing)

#creating your machine learning model object
#regr = linear_model.LinearRegression()

# This first line will work, but the 2nd performs a search for optimal hyperparameter settings for the algorithm
#regr = kernel_ridge.KernelRidge(kernel='rbf')
regr = GridSearchCV(kernel_ridge.KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

#fitting your machine learning model object
regr.fit(xtrain, ytrain)

#predicting values using your fitted model
ypred = regr.predict(xtest)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(ytest, ypred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(ytest, ypred))


sorted_inds = np.argsort(ypred, axis=0)
plt.figure()
plt.scatter(np.arange(xtest.shape[0]), ytest[sorted_inds], label="Actual Price")
plt.scatter(np.arange(xtest.shape[0]), ypred[sorted_inds], label="Predicted Price")
plt.legend(loc='best')
plt.xlabel('House #')
plt.ylabel('House Price ($1000s)')
plt.plot()
plt.show()
