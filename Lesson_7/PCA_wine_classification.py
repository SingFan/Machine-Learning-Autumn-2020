# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:39:06 2019

@author: Brian Chan
"""


# =============================================================================
# 1. Data import
# =============================================================================
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# =============================================================================
# 2. PCA
# =============================================================================

import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# =============================================================================
# 3. Visualization
# =============================================================================

import matplotlib.pyplot as plt

plt.figure()
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

X_train_std[0].dot(w)

X_train_pca = X_train_std.dot(w)
colors  = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

plt.figure()
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()


# =============================================================================
# # logistic regression
# =============================================================================

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver="liblinear", C=10**10, random_state=42)
log_reg.fit(X_train_pca, y_train)

y_proba = log_reg.predict_proba(X_train_pca)

y_pred = np.zeros((len(y_proba),3))

for i in range(0,len(y_proba)):
    y_pred[i,np.argmax(y_proba[i,:])] = 1

y_actual = np.zeros((len(y_proba),3))

for i in range(0,len(y_proba)):
    y_actual[i,y_train[i]-1] = 1

print('Sum of absolute error: ',np.sum(np.sum(np.abs(y_actual - y_pred))))
print('Number of data points in training set:', len(y_actual))
# =============================================================================
# 
# =============================================================================
X_test_pca = X_test_std.dot(w)

y_proba = log_reg.predict_proba(X_test_pca)

y_pred = np.zeros((len(y_proba),3))

for i in range(0,len(y_proba)):
    y_pred[i,np.argmax(y_proba[i,:])] = 1

y_actual = np.zeros((len(y_proba),3))

for i in range(0,len(y_proba)):
    y_actual[i,y_test[i]-1] = 1

print('Sum of absolute error: ',np.sum(np.sum(np.abs(y_actual - y_pred))))
print('Number of data points in test set:', len(y_actual))






