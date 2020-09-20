# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:40:56 2019

@author: Brian Chan
"""

import numpy as np
import pandas as pd


Data = pd.read_csv('./Data/Stocks.csv', index_col = 0, header= 0)


format = '%Y-%m-%d %H:%M:%S'
Data.index = pd.to_datetime(Data.index, format='%Y/%m/%d')

import matplotlib.pyplot as plt

plt.plot(Data.iloc[:,0], Data.iloc[:,1])
plt.scatter(Data.iloc[:,0], Data.iloc[:,1])

Return = Data.pct_change()
Return.iloc[0,:] = 0.0

plt.scatter(Return.iloc[:,0], Return.iloc[:,1])
plt.scatter(Return.iloc[:,0], Return.iloc[:,2])

print(Return.cov())
print(Return.corr())

# =============================================================================
# Spread
# =============================================================================
i = 3
j = 4

Spread = Data.iloc[:,i] - np.cov(Return.iloc[:,[i,j]].T)[0,1]/np.var(Return.iloc[:,j].T)*Data.iloc[:,j]
plt.plot(Spread)

plt.plot(Data.iloc[:,[i,j]])
plt.legend(Data.columns[[i,j]])

i = 1
j = 3

Spread = Data.iloc[:,i] - np.cov(Return.iloc[:,[i,j]].T)[0,1]/np.var(Return.iloc[:,j].T)*Data.iloc[:,j]
plt.figure()
plt.plot(Spread)

plt.figure()
plt.plot(Data.iloc[:,[i,j]])
plt.legend(Data.columns[[i,j]])

# =============================================================================
# 
# =============================================================================
Data_norm = Data/Data.iloc[0,:]
plt.plot(Data_norm)
plt.legend(Data.columns)