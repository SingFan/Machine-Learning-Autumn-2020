# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:13:04 2020

@author: chans
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_excel('yield_daily.xlsx', index_col = 'date')



data.drop(columns = ['BC_1MONTH','BC_20YEAR','BC_30YEARDISPLAY', 'BC_30YEAR'],inplace = True)

data.replace(0,np.nan, inplace = True)
data.dropna(how = 'all', inplace = True)
data.fillna(method = 'ffill', inplace = True)

data.sort_index(ascending = True, inplace = True)

# =============================================================================
# Visualize
# =============================================================================

from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0, data.shape[1], 1)
y = np.arange(0, data.shape[0], 1)

X, Y = np.meshgrid(x, y)
Z = np.array(data.iloc[::-1,:]).reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('Yield level')
ax.set_ylabel('Time')
ax.set_zlabel('Maturity')

plt.show()

# =============================================================================
# PCA
# =============================================================================


from sklearn.decomposition import PCA


pca = PCA()
pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)


plt.figure()
plt.plot(cumsum)
plt.title('Cumulative variance explained along n')
plt.show()



d = np.argmax(cumsum >= 0.95) + 1

pca       = PCA(n_components=0.9995)
x_reduced = pca.fit_transform(data)
pd_x_reduced = pd.DataFrame( x_reduced, index = data.index)

pca.n_components_
np.sum(pca.explained_variance_ratio_)

pca_weight = pca.components_


x_reduced   = pca.fit_transform(data)
x_recovered = pca.inverse_transform(x_reduced)
pd_x_recovered = pd.DataFrame( x_recovered, index = data.index)
    
plt.figure()
plt.plot(pd_x_recovered)
plt.title('Recovered from PCA components')

plt.figure()
plt.plot(data)
plt.title('Yields')



# =============================================================================
# 
# =============================================================================

plt.figure()
plt.plot(pd_x_reduced)
plt.title('PCA components')


plt.figure()
plt.plot(2*(data.iloc[:,1])-10)
plt.plot(pd_x_reduced.iloc[:,0])
plt.legend(['2 * 6M minus 10', 'PCA_1'])



plt.figure()
plt.plot(data.iloc[:,0] - data.iloc[:,5])
plt.plot(-pd_x_reduced.iloc[:,1])
plt.legend(['3M minus 5 yr', 'PCA_2'])

plt.figure()
plt.plot(-data.iloc[:,0] + data.iloc[:,3] -data.iloc[:,4] + data.iloc[:,7])
plt.plot(pd_x_reduced.iloc[:,1])
plt.legend(['(10yr - 5yr) - (2yr - 3m)', 'PCA_3'])





















