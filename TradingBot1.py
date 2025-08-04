import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient, StockTradesRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import yfinance as yf
start = datetime.datetime(2021,1,1)
end = datetime.datetime(2024,6,1)
finstocks = ['JPM', 'BAC', 'C', 'GS', 'MS', 'BLK', 'BX', 'PYPL', 'SPY']
cryptostocks = [ 'NVDA', 'MSTR', 'GBTC']
stocks =yf.download(finstocks, start, end)
stockdata = stocks.xs('Close', level = 0, axis = 1)
print(stockdata.head())
rows, cols = 9,9
cointegration = [[0 for _ in range(cols)] for i in range(rows)]
for i in range(len(finstocks)):
    for j in range(len(finstocks)):
        score, pvalue, __= coint(stockdata[finstocks[i]], stockdata[finstocks[j]])
        cointegration[i][j] = pvalue
plt.figure()

plt.imshow(cointegration, cmap = 'viridis_r')
plt.colorbar(label='Value')
plt.xticks(ticks = range(len(finstocks)),labels=finstocks)
plt.yticks(ticks = range(len(finstocks)),labels=finstocks)


stock_1 = stockdata['MS']
stock_2 = stockdata['BLK']
stockratio = stock_1/stock_2
stockmargin = stock_1 - stock_2


rolling_5_ratio= stockratio.rolling(window = 5, center = False).mean()
rolling_20_ratio = stockratio.rolling(window = 40, center = False).mean()
rolling_total = stockratio.expanding().mean()
trades = pd.DataFrame()
trades['zscore'] = ((rolling_5_ratio -rolling_total)/stockratio.std())

trades['trade'] = 0

trades = trades.reset_index()
for i in range(len(stockratio)):
    if(trades['zscore'].iloc[i]> 1):
        trades.at[i, 'trade'] = 1
    elif(trades['zscore'].iloc[i] < -1):
         trades.at[i, 'trade'] = -1
    elif((trades['zscore'].iloc[i] < .25) and (trades['zscore'].iloc[i]>-.25)):
         trades.at[i, 'trade'] = -2
buy = trades[trades['trade'] == -1 ]
sell = trades[trades['trade'] == 1 ]
off = trades[trades['trade'] == -2]

plt.figure()
plt.plot(trades.index, trades['zscore'])
plt.plot(buy.index, trades['zscore'].loc[buy.index], marker = 'v', markersize=10, color='r', label='Sell')
plt.plot(sell.index, trades['zscore'].loc[sell.index], marker = '^', markersize=10, color = 'g', label = 'Sell')
plt.figure()
plt.plot(range(len(rolling_5_ratio)), rolling_5_ratio)
plt.plot(range(len(rolling_total)), rolling_total)
plt.plot(buy.index, stockratio[buy.index], marker = 'v', markersize=10, color='r', label='Sell')
plt.plot(sell.index, stockratio[sell.index], marker = '^', markersize=10, color = 'g', label = 'Sell')
plt.plot(off.index, stockratio[off.index], marker = 'o', markersize=4, color = 'gray', label = 'Sell')



cash = 100000
cashlist = [100000]
investment = [0,0]
adj_stock1 = []
adj_stock2 = []

for i in range(len(trades)):
    multiplier = 5000/stock_1.iloc[i]
    multiplierBLK = 5000/stock_2.iloc[i]
    adj_stock1.append(stock_1.iloc[i] * multiplier)
    adj_stock2.append(stock_2.iloc[i] * multiplierBLK)
    if(trades['trade'].iloc[i] == -1):
        investment[0] += 1 * multiplier
        investment[1] -=1 * multiplierBLK
        
    if(trades['trade'].iloc[i] == 1):
        investment[0] -=1 * multiplier
        investment[1] += 1 * multiplierBLK
    if(trades['trade'].iloc[i] == -2):
        cash += investment[0] * stock_1.iloc[i] + investment[1] * stock_2.iloc[i]
        investment = [0,0]
    cashlist.append(cash + investment[0] * stock_1.iloc[i] + investment[1] * stock_2.iloc[i])
plt.figure()
plt.plot(range(len(cashlist)), cashlist)
print(cash)
plt.figure()
plt.plot(range(len(stock_1)), stock_1)
plt.plot(range(len(stock_2)), stock_2*(stock_1.iloc[0]/stock_2.iloc[0]))
plt.plot(buy.index, (stock_1[buy.index]+(stock_2[buy.index]*(stock_1.iloc[0]/stock_2.iloc[0])))/2, marker = '^', markersize=10, color = 'g', label = 'Sell')
plt.plot(sell.index, (stock_1[sell.index]+(stock_2[sell.index]*(stock_1.iloc[0]/stock_2.iloc[0])))/2, marker = 'v', markersize=10, color = 'r', label = 'Sell')
plt.plot(off.index, (stock_1[off.index]+(stock_2[off.index]*(stock_1.iloc[0]/stock_2.iloc[0])))/2, marker = 'o', markersize=4, color = 'gray', label = 'Sell')
plt.show()
