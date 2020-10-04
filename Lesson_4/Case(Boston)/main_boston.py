# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:17:49 2019

@author: Brian Chan
"""

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

#%matplotlib inline

# =============================================================================
# Definition of variables
# =============================================================================
# target variable
# MEDV - Median value of owner-occupied homes in $1000's

# explanatory variables

#CRIM: Per capita crime rate by town
#ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
#INDUS: Proportion of non-retail business acres per town
#CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#NOX: Nitric oxide concentration (parts per 10 million)
#RM: Average number of rooms per dwelling
#AGE: Proportion of owner-occupied units built prior to 1940
#DIS: Weighted distances to five Boston employment centers
#RAD: Index of accessibility to radial highways
#TAX: Full-value property tax rate per $10,000
#PTRATIO: Pupil-teacher ratio by town
#B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
#LSTAT: Percentage of lower status of the population
#MEDV: Median value of owner-occupied homes in $1000s


#Load the Boston Housing DataSet from scikit-learn
from sklearn.datasets import load_boston

boston_dataset = load_boston()

# boston_dataset is a dictionary
boston_dataset.keys() # check the content

#Load the data into pandas dataframe
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()

#The target values is missing from the data. Create a new column of target values and add it to dataframe
boston['MEDV'] = boston_dataset.target

#Data preprocessing
# check for missing values in all the columns
boston.isnull().sum()
# =============================================================================
# Data Visualization
# =============================================================================
# # plot a histogram of the target values
# plt.figure()
# sns.set(rc={'figure.figsize':(11.7,8.27)}) # set the size of the figure
# sns.distplot(boston['MEDV'], bins=30)
# plt.show()

#plt.figure()
#plt.hist(boston['MEDV'], bins=30)

# =============================================================================
# Correlation matrix
# =============================================================================
# # compute the pair wise correlation for all columns  
# correlation_matrix = boston.corr().round(2)

# # use the heatmap function from seaborn to show the correlation matrix
# # annot = True to print the values inside the square
# plt.figure()
# sns.heatmap(data=correlation_matrix, annot=True, cmap="YlGnBu") # , cmap="YlGnBu" , cmap="Blues" , cmap="BuPu"

# plt.figure(figsize=(20, 5))

# features = ['LSTAT', 'RM']
# target   = boston['MEDV']

# for i, col in enumerate(features):
#     plt.subplot(1, len(features) , i+1)
#     x = boston[col]
#     y = target
#     plt.scatter(x, y, marker='o')
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel('MEDV')

# =============================================================================
# Prepare the data for training
# =============================================================================
#X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
#Y = boston['MEDV']

X = boston.iloc[:,:-1]
Y = boston['MEDV']

# =============================================================================
# Split the data into training and testing sets
# =============================================================================

from sklearn.model_selection import train_test_split

# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# normalizing the variables
X_test  = (X_test  - X_train.mean())/X_train.std()
X_train = (X_train - X_train.mean())/X_train.std()

# =============================================================================
# Train the model using sklearn LinearRegression
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

from sklearn import linear_model
lin_model_lasso = linear_model.Lasso(alpha=0.07)
lin_model_lasso.fit(X_train, Y_train)

print(lin_model_lasso.coef_)


#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# =============================================================================
# # model evaluation for Linear regression
# =============================================================================

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2   = r2_score(Y_train, y_train_predict)


print("=======================================")
print("           Linear regression          ")
print("=======================================")

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is     {}'.format(rmse))
print('R2 score is {}'.format(r2))


# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is     {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("--------------------------------------")
print("\n")
# plotting the y_test vs y_pred
# ideally should have been a straight line
plt.figure()
plt.scatter(Y_test, y_test_predict)
plt.show()

#plt.figure()
#plt.plot(np.array(Y_test))
#plt.plot(np.array(y_test_predict))
#plt.show()


# =============================================================================
# # model evaluation for Lasso regression
# =============================================================================

y_train_predict = lin_model_lasso.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2   = r2_score(Y_train, y_train_predict)

print("=======================================")
print("       Linear regression (Lasso)      ")
print("=======================================")
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is     {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict = lin_model_lasso.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2   = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is     {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("--------------------------------------")
print("\n")

















