# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:00:56 2019

@author: Brian Chan
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
data = pd.read_csv('data_spurious.txt', header = None) # US spend, suicides

Y = np.reshape(np.array(data.iloc[:,0]),(-1,1))
X = np.reshape(np.array(data.iloc[:,1]),(-1,1))

plt.figure()
plt.scatter(Y,X)
plt.title('Scatter plot of Y against X')

# =============================================================================
# Package: sklearn 
# =============================================================================

lin_model = LinearRegression()
lin_model.fit(X, Y)

#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# model evaluation for training set
Y_pred = lin_model.predict(X)
rmse   = (np.sqrt(mean_squared_error(Y, Y_pred)))
r2     = r2_score(Y, Y_pred)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

plt.figure()
plt.plot(Y)
plt.plot(Y_pred)
plt.title('Plot of $Y$ against $\hat Y$')


# plotting the y_test vs y_pred
# ideally should have been a straight line
#plt.figure()
#plt.scatter(Y, Y_pred)
#plt.title('Scatter plot of Y against Y_pred')

# =============================================================================
# Package: statsmodels
# =============================================================================
import statsmodels.api as sm

data = pd.read_csv('data_spurious.txt', header = None) # US spend, suicides

Y = np.reshape(np.array(data.iloc[:,0]),(-1,1))
X = np.reshape(np.array(data.iloc[:,1]),(-1,1))

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

















