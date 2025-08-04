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
trade_client = TradingClient('PK22TRGUEJF8RFFYNOSI','m1hicdWaHlI9dvnv8woE3zSTAL59r9a5gqVWhfCM')
data_client = StockHistoricalDataClient('PK22TRGUEJF8RFFYNOSI','m1hicdWaHlI9dvnv8woE3zSTAL59r9a5gqVWhfCM')
print(trade_client.get_account().account_number)
print(trade_client.get_account().buying_power)
aapl = StockBarsRequest(symbol_or_symbols=['MS'],timeframe=TimeFrame.Day, start=datetime.datetime(2024,6,1), end=datetime.datetime(2025,8,1),adjustment="split")
msft = StockBarsRequest(symbol_or_symbols=['BX'],timeframe=TimeFrame.Day, start=datetime.datetime(2024,6,1), end=datetime.datetime(2025,8,1),adjustment="split")
msftbars = data_client.get_stock_bars(msft)
msftbars = msftbars.df.reset_index()
bars = data_client.get_stock_bars(aapl)
bars = bars.df.reset_index()

diff = (bars['close'] / msftbars['close']).rolling(window=100).mean()
spread = (bars['close'] -(msftbars['close']* diff))
priceratio = bars['close']/(msftbars['close'] * diff)
barscut = bars[100:]
msftbarscut = msftbars[100:]
diffcut = diff[100:]
score, pvalue, __ = coint(bars['close'], msftbars['close'])
print(pvalue)
plt.figure(figsize=(10,8))

plt.plot(barscut['timestamp'], barscut['close'])

plt.plot(msftbarscut['timestamp'], msftbarscut['close'] )
plt.plot(bars['timestamp'], spread)
plt.legend('Bars','MSFT')

plt.axhline(y=0)


plt.figure()
plt.plot(bars['timestamp'], priceratio)

plt.figure()

df_zscore = (priceratio-priceratio.mean())/priceratio.std()
plt.plot(df_zscore)
plt.axhline(1, color = 'green')
plt.axhline(1.25, color = 'green')
plt.axhline(-1, color = 'red')
plt.axhline(-1.25, color = 'red')

plt.figure()
ratios_mavg5 = priceratio.rolling(window = 5, center = False).mean()
ratios_mavg20 = priceratio.rolling(window=25, center=False).mean()
std_20 = priceratio.rolling(window=25, center=False).std()
zscore_20_5 = (ratios_mavg5-ratios_mavg20)/std_20
plt.plot(priceratio.index, priceratio)
plt.plot(ratios_mavg5.index,ratios_mavg20.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.legend(['Ratio', "5D Ratio",'20D Ratio'])


plt.figure()
plt.plot(zscore_20_5.index,zscore_20_5.values)


signals = pd.DataFrame(index=priceratio.index)
signals['zscore'] = zscore_20_5
signals['signal'] = 0
signals.loc[signals['zscore'] < -.75, 'signal'] = 1
signals.loc[signals['zscore'] > .75, 'signal'] = -1
signals.loc[signals['zscore'].between(-.1, .1), 'signal'] = -2

print(signals.head(100))
plt.figure(figsize=(12,6))
plt.plot(priceratio.index, priceratio, label='Price Ratio')
plt.plot(ratios_mavg5.index, ratios_mavg5.values, label='5-day MA')
plt.plot(ratios_mavg20.index, ratios_mavg20.values, label='20-day MA')

# Plot Buy signals
buy_signals = signals[signals['signal'] == 1]
plt.plot(buy_signals.index, priceratio[buy_signals.index], '^', markersize=10, color='g', label='Buy')

# Plot Sell signals
sell_signals = signals[signals['signal'] == -1]
plt.plot(sell_signals.index, priceratio[sell_signals.index], 'v', markersize=10, color='r', label='Sell')

plt.legend()
plt.title("Trading Signals Based on Price Ratio")


plt.figure(figsize=(14,6))
plt.plot(bars['timestamp'], bars['close'], label='AAPL Price', alpha=0.7)
plt.plot(msftbars['timestamp'], msftbars['close'] * diff, label='MSFT Price', alpha=0.7)

initial_cash = 100000
cash = initial_cash
position = 0
multi = 50
portfolio_values = [0,0]
cash_list = []
signals = signals.reset_index()
for i, row in signals.iterrows():
    aapl_price = bars['close'].iloc[i]
    msft_price = msftbars['close'].iloc[i]
    pricedif = aapl_price/msft_price
    if row['signal'] == -1:
        cash += aapl_price * multi
        portfolio_values[0] -= 1 * multi
        if(cash >= msft_price * multi):
            cash -= msft_price* multi * pricedif
            portfolio_values[1] += 1* multi * pricedif
    if row['signal'] == 1:
        cash += msft_price* multi * pricedif
        portfolio_values[1] -=1* multi * pricedif
        if(cash >= aapl_price * multi):
            cash -= aapl_price* multi
            portfolio_values[0] +=1* multi
    if row['signal'] == -2:
        cash += portfolio_values[0] * aapl_price + portfolio_values[1] * msft_price
        portfolio_values = [0,0]
    cash_list.append(cash + portfolio_values[0] * bars['close'].iloc[-1] + portfolio_values[1] * msftbars['close'].iloc[-1])
final_value = cash + portfolio_values[0] * bars['close'].iloc[-1] + portfolio_values[1] * msftbars['close'].iloc[-1]
print(f"Total End Value is ${final_value:,.2f}")
plt.figure()
plt.plot(range(len(cash_list)), cash_list)
plt.show()


# Download JPM and MS price data from 2015 to 2025
jpm_df = yf.download('SPY', start='2022-01-01', end='2025-01-01')
ms_df = yf.download('BRK-B', start='2022-04-01', end='2025-01-01')

# Flatten just in case the column names are multi-level
jpm = jpm_df['Close'].dropna()
ms = ms_df['Close'].dropna()
jpm = jpm.squeeze()
ms = ms.squeeze()
print("Length of JPM:", len(jpm))
print("Length of MS:", len(ms))
print(jpm.shape, ms.shape)
print(jpm.head())
print(ms.head())
print(jpm.index.equals(ms.index))
# Align the time series
jpm, ms = jpm.align(ms, join='inner')
# Run the Engle-Granger cointegration test
score, pvalue, _ = coint(jpm, ms)
correlation = jpm.corr(ms)
print(f"Pearson correlation coefficient between JPM and MS: {correlation:.4f}")
print("Test statistic:", score)
print("P-value:", pvalue)