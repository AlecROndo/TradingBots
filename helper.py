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
from sklearn.linear_model import LinearRegression

#Trying Linear Regression
def lreg(entries, stockrat):
    regdata = pd.DataFrame(columns = ['coef', 'intercept', 'value'])
    for i in range(0, len(entries)):
        if i == 0:  # Only one point, can't fit regression
            regdata.loc[i] = [0, stockrat[i], stockrat[i]]
        elif i<252:
            tempentries = np.array(entries[:i]).reshape(-1,1)
            tempstockrat = np.array(stockrat[:i])
            model = LinearRegression()
            model.fit(tempentries, tempstockrat)
            regdata.loc[i] = [model.coef_[0], model.intercept_, model.coef_[0] * i + model.intercept_]
        else: 
            tempentries = np.array(entries[i-252:i]).reshape(-1,1)
            tempstockrat = np.array(stockrat[i-252:i])
            model = LinearRegression()
            model.fit(tempentries, tempstockrat)
            regdata.loc[i] = [model.coef_[0], model.intercept_, model.coef_[0] * i + model.intercept_]
    
    return regdata

    