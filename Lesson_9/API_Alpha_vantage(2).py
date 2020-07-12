# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 00:00:51 2019

@author: Brian Chan
"""

from alphavantage.price_history import (
  AdjustedPriceHistory, get_results, PriceHistory, IntradayPriceHistory,
  filter_dividends
)

## weekly prices
#history = PriceHistory(period='W', output_size='compact')
#results = history.get('AAPL')
#
## intraday prices, 5 minute interval
#history = IntradayPriceHistory(utc=True, interval=5)
#results = history.get('AAPL')
#
## adjusted daily prices
#history = AdjustedPriceHistory(period='D')
#results = history.get('AAPL')
#dividends = list(filter_dividends(results.records))
#
## Return multiple tickers
#parameters = {'output_size': 'compact', 'period': 'D'}
#tickers = ['AAPL', 'MSFT']
#results = dict(get_results(PriceHistory, tickers, parameters))


from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

ts = TimeSeries(key='YOUR OWN KEY', output_format='pandas')

ts = TimeSeries(key='ORVJZ8BPU84FLDNN', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
data['4. close'].plot()
plt.title('Intraday Times Series for the MSFT stock (1 min)')
plt.show()


#Ticker = list(['BA','LMT','GD','RTN','NOC','TDG','MSI','LLL','TDY','HII','FLIR','CW','AAXN','MOG.A','AJRD','ESE'
#              ,'CUB','MANT','KAWN','AVAV','TGI','AIR'])
Ticker = list(['BA','LMT','GD','RTN','NOC','TDG','MSI','LLL','TDY','HII','FLIR','CW','AAXN','AJRD','ESE'
              ,'CUB','MANT','KAWN','AVAV','TGI','AIR'])

ts = TimeSeries(key='YOUR OWN KEY', output_format='pandas')

Data      = dict()
Meta_data = dict()

import time
for i, ticker in enumerate(Ticker):
    print(ticker)
    try:
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full') #'full' 'compact'
    except:
        time.sleep(5)
    
    Data.update({ticker:data})
    Meta_data.update({ticker:meta_data})

#data['close'].plot()

# =============================================================================
# 
# =============================================================================

import pandas as pd

Price = pd.DataFrame([], index = Data['AAXN'].index[-4430:])

for ticker in Data:
    print(ticker)
    print(Data[ticker].iloc[-4430:,4].head())
    Price[ticker] = Data[ticker].iloc[-4430:,4]










