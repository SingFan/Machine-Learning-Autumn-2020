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

ts = TimeSeries(key='ORVJZ8BPU84FLDNN', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

# today       = dt.date.today()
# thirty_days = dt.timedelta(days=30)
# thirty_days_ago = today-thirty_days
# data = quandl.get("WIKI/AAPL", start_date=str(thirty_days_ago), end_date=str(today))
# data.plot();
# plt.show()

# quandl.get("WIKI/AMZN", start_date="2017-7-10", end_date="2018-7-10")

import pandas as pd

Ticker = pd.read_csv('tickers_sp500.csv', header = 0)

Dataset = pd.DataFrame()
Dataset_dict      = dict()
Dataset_info_dict = dict()
for ticker in Ticker['Symbol']:
    try:
        data, meta_data = ts.get_intraday(symbol=ticker,interval='5min', outputsize='full')
        
        Dataset_dict[ticker]      = data
        Dataset_info_dict[ticker] = meta_data
        
        # Dataset_dict[ticker] = quandl.get("WIKI/"+ticker, start_date="2000-1-1", end_date="2018-12-31")
        # Dataset_dict[ticker] = quandl.get("WIKI/"+ticker, start_date="2000-1-1", end_date="2018-12-31")

        print(ticker, 'Data length:', len(Dataset_dict[ticker]['close']))
    except:
        print(ticker, 'error')

for ticker in Ticker['Symbol']:
    print(ticker)
    # Dataset_dict[ticker] = quandl.get("WIKI/"+ticker, start_date="2000-1-1", end_date="2018-12-31")
    try:
        Dataset[ticker] = Dataset_dict[ticker]['close']
    # print('Data length: ', len(Dataset_dict[ticker]['Adj. Close']))
    except:
        pass


Dataset.to_csv('Data_sp500.csv')

with open('Data_sp500.p', 'wb') as fp:
    pickle.dump(Dataset_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)



# =============================================================================
# Eye-ball check
# =============================================================================
# MMM delete the last segment
# BKNG delete whole series
# =============================================================================
# 
# =============================================================================
    
    
    
    
    
# # with open('Data_sp100.p', 'rb') as fp:
# #     data = pickle.load(fp) 
# # =============================================================================
# # organizing data
# # =============================================================================
# count_null = (~Dataset.isnull()).sum()

# selected_tickers = count_null.index[count_null>3250]
# Dataset_selected = Dataset[selected_tickers]

# Dataset_selected.to_csv('Data_sp100_selected.csv')

# # =============================================================================
# # Commodity
# # =============================================================================
# Ticker_commodity = ['BP/SPOT_CRUDE_OIL_PRICES','WGC/GOLD_DAILY_USD','BCIW/_INX']

# Dataset = pd.DataFrame()
# Dataset_dict = dict()


# ticker = 'BP/SPOT_CRUDE_OIL_PRICES'
# Dataset_dict[ticker] = quandl.get(ticker, start_date="2005-1-1", end_date="2018-12-31")
# Dataset[ticker] = Dataset_dict[ticker]['Brent']

# ticker = 'WGC/GOLD_DAILY_USD'
# Dataset_dict[ticker] = quandl.get(ticker, start_date="2005-1-1", end_date="2018-12-31")
# Dataset[ticker] = Dataset_dict[ticker]['Value']

# Ticker_commodity = 'BCIW/_INX'
# Dataset_dict[ticker] = quandl.get(ticker, start_date="2005-1-1", end_date="2018-12-31")
# Dataset[ticker] = Dataset_dict[ticker]['Value']




# Dataset.to_csv('Data_commodity.csv')

# with open('Data_commodity.p', 'wb') as fp:
#     pickle.dump(Dataset_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

# https://fred.stlouisfed.org/series/DCOILBRENTEU

# https://www.investing.com/commodities/gold-historical-data









