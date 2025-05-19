import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.stats import rankdata


def bbwp(data: pd.Series, length:int ,lookback:int, ma_type: str):
    data_ = pd.DataFrame({'source': data.values})

    try:
        if ma_type == 'sma':
            data_['ma'] = ta.sma(data_['source'], length= length)
        elif ma_type == 'ema':
            data_['ma'] = ta.ema(data_['source'], length = length)
        elif ma_type == 'hma':
            data_['ma'] = ta.hma(data_['source'], length = length)
        elif ma_type == 'wma':
            data_['ma'] = ta.wma(data_['source'], length= length)
        elif ma_type == 'rma':
            data_['ma'] = ta.rma(data_['source'], length = length)
    except:
        raise ValueError("If you are passing a 'ma_type' parameter accepted arguements are ['sma', 'ema', 'hma', 'wma', 'rma']")
    
    data_['sd'] = data_['source'].rolling(length).std()
    data_['bbw'] = (data_['ma'] + data_['sd'] - (data_['ma'] - data_['sd']) ) / data_['ma']

    try:
        # Calculate BBWP (Bollinger Band Width Percent)
        data_['bbw diff'] = (data_['bbw'].diff().fillna(0) >= 0).astype(int)
        data_["bbw_sum"] = data_["bbw diff"].rolling(lookback, min_periods= 1).sum()
        
        # Rolling sum of 'bbw increases'
        data_['bbwp'] = data_["bbw_sum"].rolling(lookback, min_periods= lookback).apply(lambda x: rankdata(x)[-1] / len(x) * 100, raw = False)

    except:
        raise ValueError(f"With {len(data)} data points you can't calculate BBWP with {lookback} lookback")
        
    return data_['bbwp'].values