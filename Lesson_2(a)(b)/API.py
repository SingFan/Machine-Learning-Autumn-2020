# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:10:46 2020

@author: Brian Chan
"""

import quandl
import datetime as dt
import matplotlib.pyplot as plt
import pickle
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time

# =============================================================================
# Download intraday data
# =============================================================================

ts = TimeSeries(key='ORVJZ8BPU84FLDNN', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
ts.get_daily_adjusted(symbol='MSFT', outputsize='full')

# today       = dt.date.today()
# thirty_days = dt.timedelta(days=30)
# thirty_days_ago = today-thirty_days
# data = quandl.get("WIKI/AAPL", start_date=str(thirty_days_ago), end_date=str(today))
# data.plot();
# plt.show()

# quandl.get("WIKI/AMZN", start_date="2017-7-10", end_date="2018-7-10")

# =============================================================================
# Import list of tickers
# =============================================================================
Data_ticker = pd.read_excel('Ticker_list.xlsx', header = 0)
Data_ticker = Data_ticker.iloc[:5,:]

# =============================================================================
# Calling API
# =============================================================================
Dataset = pd.DataFrame()
Dataset_dict      = dict()
Dataset_info_dict = dict()
for ticker in Data_ticker['Ticker']:
    try:
        time.sleep(12) # avoid exceed calling limit (5 asks per minute)
        # data, meta_data = ts.get_intraday(symbol=ticker,interval='5min', outputsize='full')
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        Dataset_dict[ticker]      = data
        Dataset_info_dict[ticker] = meta_data

        print(ticker, 'Data length:', len(Dataset_dict[ticker]['5. adjusted close']))
    except:
        print(ticker, 'error')

for ticker in Data_ticker['Ticker']:
    print(ticker)
    try:
        Dataset[ticker] = Dataset_dict[ticker]['5. adjusted close']
    except:
        pass

Dataset.sort_index(ascending=True, inplace = True)
Dataset.to_csv('Data_sp500.csv')

with open('Data_sp500.p', 'wb') as fp:
    pickle.dump(Dataset_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

# with open('Data_sp500.p', 'rb') as fp:
#     data2 = pickle.load(fp) 

# =============================================================================
# Pre-processing the data
# =============================================================================
Dataset = pd.read_csv('Data_sp500.csv', index_col = 0)

Dataset.iloc[5,0] = np.nan
Dataset.fillna(method = 'ffill', inplace = True)
Dataset.fillna(method = 'bfill', inplace = True)


# =============================================================================
# Basic tranformations
# =============================================================================

# Daily return
Return = Dataset.pct_change(1)
Return.iloc[0,:] = 0

# Moving average
horizon = 12
MA = Dataset.rolling(window = horizon).mean()
MA.fillna(method = 'bfill', inplace = True)

# Check for abnormal columns 
# (e.g. look at std of the columns)
Return.describe()
Return.std()














