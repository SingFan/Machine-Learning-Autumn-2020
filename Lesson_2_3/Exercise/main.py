# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:19:12 2019

@author: Brian Chan
"""

# =============================================================================
# Exercise: Ticker matching
# =============================================================================

import numpy as np
import pandas as pd

Tickers = pd.read_excel('Ticker_list.xlsx', header = None)

Data = pd.read_excel('data.xlsx', index_col = 0, header = 0)

Data_selected = Data[[Tickers.iloc[i,0] for i in np.arange(len(Tickers)) ]]

import matplotlib.pyplot as plt

plt.plot(Data_selected)
plt.gca().legend([Tickers.iloc[i,0] for i in np.arange(len(Tickers)) ])


