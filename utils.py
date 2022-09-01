import pandas as pd
import numpy as np
import scipy.stats as stats
import operator
from collections import OrderedDict
import random
import yfinance as yf


def sma(data_levels,params = 50):
    
    data = pd.DataFrame(data_levels).copy()
    
    data['SMA_{}'.format(params)]  = data.iloc[:,0].rolling(params).mean()
    data['Returns'] = data.iloc[:,0].pct_change()
    data = data.dropna()

    signals = []

    for i in range(len(data)):
        
        ## if SMA > current price = buy
        if data.iloc[:,0].iloc[i] > data.iloc[:,1].iloc[i]:
            signals.append(1)
        
        # if SMA < current price = sell
        if data.iloc[:,0].iloc[i] < data.iloc[:,1].iloc[i]:
            signals.append(0)
            
        # if SMA == current price = do nothing (this wont happen)
        if data.iloc[:,0].iloc[i] == data.iloc[:,1].iloc[i]:
            print('test')
            signals.append(0)
            
    data['Signals'] = signals
    
    return data




def backtester(signals,price, tcost = 0.005):

        pos_val = np.zeros(np.shape(price))
        cash    = np.zeros(np.shape(price))
        cash[0] = 1

        for i,val in enumerate(price):

            if i == len(price)-1:
                break


            if signals[i] == 0:

                cash[i+1] = (pos_val[i] * val * (1-tcost)) + cash[i]
                pos_val[i+1] = 0

            elif signals[i] == 1:

                pos_val[i+1] = (cash[i] / val)*((1-tcost)) + pos_val[i]
                cash[i+1] = 0


        returns = [a*b for a,b in zip(pos_val,price)] + cash
        
        return pd.DataFrame(returns, index = price.index)
    
    

def win_rate(sigs, returns):

    """
    Signals : series
    returns : series
    must be same length
    """

    tps = []

    sigs = sigs[1:-1].values.ravel()

    rets = (returns.pct_change()).shift(1).dropna().values.ravel()

    for i,val in enumerate(sigs):

        if (sigs[i] == 1 and rets[i]>0):
            tps.append(1)

    win_rate = sum(tps)/len(sigs)
    return win_rate



def max_dd(returns, window=None):

    data = (1+pd.Series(returns)).cumprod()

    if window is not None:
        roll_max = data.rolling(window, min_periods=1).max()
    else:
        roll_max = data.expanding().max()

    daily = data/roll_max - 1.0

    return min(daily)



def sharpe(data, n=365):

    return data.mean() / data.std() * np.sqrt(n)


def sharpe_t_test(data, n=365):
    """
    Campbell Harvey Bactesting
    """
    t_stat= abs(sharpe(data)*np.sqrt(len(data)/n))       
    pval = stats.t.sf(np.abs(t_stat), (len(data))-1)*2  
    return pval


def mtt_bon(dictionary, n_strats = 3):
    """
    takes a dictionary of p values, use name of the strat as the keys
    returns an ordered list of the p-values
    """
    dict_1 = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
    dict_1.update((x, y*n_strats) for x, y in dict_1.items())
    p_val_bon = {k: v for k, v in sorted(dict_1.items(), key=lambda item: item[1])}
    return p_val_bon