
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import GaussianNoise
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import concatenate
from keras import regularizers
import keras
import pandas as pd
import numpy as np
import time
import random
import math
import optuna
import plotly
from keras.backend import clear_session
from optuna.visualization import plot_contour
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.visualization import plot_optimization_history
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState


PERCENT_CHANGE = False   
WINDOW = 20
STEP = 2

COMMISSION_FRAC = 0.001
COMMISSION_MULT = 5  
ATR_MULT = 0.25

BATCH_SIZE = 256
RANDOM_SEED = 7
FILTERS = 512
NOISE_SDEV = 0.00
DEEP_NOISE_SDEV = 0.00
PATIENCE = 2


def objective(trial):

    clear_session()

    PERIOD = trial.suggest_int("period", 25, 40)                 
    PERIOD_FAST = trial.suggest_int("period_fast", 7, 15)      
    ISHIMOKU_SCALE = trial.suggest_int("ishi_scale", 3, 7)

    # Create categorical labels.

    def convert_unix_time_into_day(seconds):
        """ returns day of week as category label """
        day_of_week = time.strftime('%A', time.localtime(seconds))
        dict = {'Saturday':1, 'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday':5, 'Thursday':6, 'Friday':7}
        day_code = dict[day_of_week]
        return day_code


    def wwma(pd_series, period):
        """ w. wilder's exponential moving average """
        return pd_series.ewm(alpha=1/period, adjust=False, ignore_na=True).mean()


    def atr(df, length=14):
        """ average true range (for column with latest values at top) """
        df_high, df_low, df_prev_close = df['high'], df['low'], df['close'].shift()
        df_tr = [df_high - df_low, df_high - df_prev_close, df_low - df_prev_close]
        df_tr = [tr.abs() for tr in df_tr]
        df_tr = pd.concat(df_tr, axis=1).max(axis=1)
        df_atr = wwma(df_tr, length)
        return df_atr


    # load price data into dataframe and reorder to most recent at bottom
    df = pd.read_csv('data/OHLC_1h.csv', header=0, sep=',')
    df = df.sort_index(ascending=False, ignore_index=True)

    # calculate average true range
    df_atr = atr(df)
    df['ATR'] = df_atr

    # create target labels 
    y_labels =[]

    for i in range(2,len(df)):

        # up condition: (high or high[1] > close[2] + spread) and (low and low[1] > low[2] + spread)     
        if (
            ( df.loc[i, 'high'] > (df.loc[i-2, 'close'] + ATR_MULT * df.loc[i, 'ATR'])  or  
            df.loc[i-1, 'high'] > (df.loc[i-2, 'close'] + ATR_MULT * df.loc[i, 'ATR']) )  and 

            ( df.loc[i, 'high'] > (df.loc[i-2, 'close'] * (1 + COMMISSION_FRAC * COMMISSION_MULT))  or  
            df.loc[i-1, 'high'] > (df.loc[i-2, 'close'] * (1 + COMMISSION_FRAC * COMMISSION_MULT)) )  and

            ( df.loc[i, 'low'] > (df.loc[i-2, 'low'] + ATR_MULT * df.loc[i, 'ATR'])  and  
            df.loc[i-1, 'low'] > (df.loc[i-2, 'low'] + ATR_MULT * df.loc[i, 'ATR']) )
        ):
            label = 'up'

        # down condition: (low or low[1] < close[2] - spread) and (high and high[1] < high[2] - spread)    
        elif (
            ( df.loc[i, 'low'] < (df.loc[i-2, 'close'] - ATR_MULT * df.loc[i, 'ATR'])  or  
            df.loc[i-1, 'low'] < (df.loc[i-2, 'close'] - ATR_MULT * df.loc[i, 'ATR']) )  and 
            
            ( df.loc[i, 'low'] < (df.loc[i-2, 'close'] * (1 - COMMISSION_FRAC * COMMISSION_MULT))  or  
            df.loc[i-1, 'low'] < (df.loc[i-2, 'close'] * (1 - COMMISSION_FRAC * COMMISSION_MULT)) )  and

            ( df.loc[i, 'high'] < (df.loc[i-2, 'high'] - ATR_MULT * df.loc[i, 'ATR'])  and  
            df.loc[i-1, 'high'] < (df.loc[i-2, 'high'] - ATR_MULT * df.loc[i, 'ATR']) )
        ):
            label = 'down'
        
        else:
            label = 'flat'

        y_labels.append(label)

    y_labels.extend([float('NaN'), float('NaN')])

    # add labels to new dataframe column
    df['target'] = y_labels

    # create day and hour categories
    df['day'] = df['unix'].map(lambda x: convert_unix_time_into_day(x))
    df['hour'] = df['date'].str.slice(start=11, stop=13).apply(pd.to_numeric)
    df['year'] = df['date'].str.slice(start=0, stop=4).apply(pd.to_numeric)

    pd.set_option("expand_frame_repr", False)
    pd.set_option("display.max_columns", 100)

    df_ylabels = df['target'].copy()
    df_ylabels = pd.get_dummies(df_ylabels, columns=['target'])
    del df_ylabels


    # Create features using variance, filters, significant levels, volume indicators, and fractal dimension

    def bollinger_k(df, length=20):
        """ bollinger coeffient and delta standard deviation"""
        df_tp = (df['high'] + df['low'] + df['close']) / 3
        df_sma = df_tp.rolling(length).mean()
        df_sdev = df_tp.rolling(length).std()
        df_delta_sdev = df_sdev.diff()
        df_bk = (df['close'] - df_sma) / df_sdev 
        return df_bk, df_delta_sdev


    def rr_vwap(dataframe, length):
        """ relative rolling volume-weighted average price """
        df = dataframe.copy()
        df['period_total'] = (df['high'] + df['low'] + df['close'] / 3) * df['volume']
        df['cum_tot'] = df['period_total'].rolling(length).sum()
        df['cum_vol'] = df['volume'].rolling(length).sum()
        df_rvwap = df['close'] - (df['cum_tot'] / df['cum_vol']) / df['close']
        return df_rvwap


    # @numba.jit
    def frama(dataframe, batch=10):
        """ fractal adaptive moving average """
        df = dataframe.copy() 
        price = df.close  
        fractal_dims = []
        filtered_prices = np.array(price)

        for i in range(0, len(df)):  
            
            if i < 2 * batch:  
                fractal_dims.append(np.nan)
                continue  

            v1 = price[i-2*batch : i - batch]  
            v2 = price[i - batch : i]
            H1 = np.max(v1)  
            L1 = np.min(v1)  
            N1 = (H1 - L1) / batch
            H2 = np.max(v2)  
            L2 = np.min(v2)  
            N2 = (H2 - L2) / batch
            H = np.max([H1, H2])  
            L = np.min([L1, L2])  
            N3 = (H - L) / (2 * batch)

            fractal_dim = 0  
            if N1 > 0 and N2 > 0 and N3 > 0:  
                fractal_dim = (np.log(N1 + N2) - np.log(N3)) / np.log(2)
            fractal_dims.append(fractal_dim)

            alpha = np.exp(-4.6 * (fractal_dim - 1))    # 4.6 limits slow EMA length < 200 
            alpha = np.max([alpha, 0.1])  
            alpha = np.min([alpha, 1])

            filtered_prices[i] = alpha * price[i] + (1 - alpha) * filtered_prices[i-1]    # single-pole lowpass filter

        df['frama'] = filtered_prices
        df['d_frama'] = df['frama'] - df['frama'].shift(1)
        df['rel_frama'] = df['frama'] / df['close']
        df['frac_dim'] = fractal_dims
        df['d_frac_dim'] = df['frac_dim'] - df['frac_dim'].shift(1)

        return df.d_frama, df.rel_frama, df.frac_dim, df.d_frac_dim


    # @numba.jit
    def z_transformer(dataframe, mode, period):
        """ digital signal processor based on a generalised z-domain transfer function """
        df = dataframe.copy() 
        price = df.close  
        transformed_prices = np.array(price)

        pi = 3.14159
        c1, N, b1, b2, a1, a2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        b0 = 1.0

        if mode == 'highpass':
            threshold_period = period
            alpha = (math.cos(2.0*pi/threshold_period) + math.sin(2.0*pi/threshold_period) - 1) / math.cos(2.0*pi/threshold_period)
            c0 = 1.0 - alpha / 2.0
            b1 = -1.0
            a1 = 1.0 - alpha

        if mode == 'bandpass':
            centre_period = period
            delta = 0.2    # half-bandwidth fraction: min=0.05, max=0.5
            beta = math.cos(2.0*pi / centre_period)
            gamma = 1.0 / math.cos(4.0*pi*delta / centre_period)
            alpha = gamma - math.sqrt(gamma**2 - 1.0)
            c0 = (1.0 - alpha) / 2.0
            b2 = -1.0
            a1 = beta * (alpha + 1.0)
            a2 = -alpha

        for i in range(0, len(df)):  
            if i < 2:  
                continue  

            transformed_prices[i] = c0 * (b0*price[i] + b1*price[i-1] + b2*price[i-2]) + a1*transformed_prices[i-1] + a2*transformed_prices[i-2] # - c1*price[N]

        df['output'] = transformed_prices
        df['delta_output'] = df.output - df.output.shift(1)
        df['relative_output'] = df.output / df.close

        return df.output, df.delta_output, df.relative_output


    def ema(pd_series, period):
        """ exponential moving average """
        return pd_series.ewm(alpha=1/period, adjust=False, ignore_na=True).mean()


    # @numba.jit
    def rma(x, n):
        """ running moving average """
        a = np.full_like(x, np.nan)
        a[n] = x[1:n+1].mean()
        for i in range(n+1, len(x)):
            a[i] = (a[i-1] * (n - 1) + x[i]) / n
        return a


    def detrended_rsi(dataframe, hp_threshold=8, rsi_period=14):
        """ relative strength index of high-frequency component of price """
        df = dataframe.copy()

        df['detrended_price'] = z_transformer(df, mode='highpass', period=hp_threshold)[0]
        df['change'] = df['detrended_price'].diff()
        df['gain'] = df.change.mask(df.change < 0, 0.0)
        df['loss'] = -df.change.mask(df.change > 0, -0.0)

        df['avg_gain'] = rma(df.gain.to_numpy(), rsi_period)
        df['avg_loss'] = rma(df.loss.to_numpy(), rsi_period)

        df['rs'] = df.avg_gain / df.avg_loss
        df['rsi'] = 100 - (100 / (1 + df.rs))

        return df.rsi


    def acc_dis_mfv(dataframe):
        """ accumulation-distribution indicator """
        df = dataframe.copy()
        price = df.close  
        acc_dis = np.array(price)
        df['mfv'] = df.volume * ((df.close - df.low) - (df.high - df.close)) / (df.high - df.low + 1)
        mfv = df['mfv'].to_numpy()

        for i in range(0, len(df)):  
            if i < 1:  
                continue  

            # if acc_dis[i-1] == np.nan:
            #     acc_dis[i-1] = price[i-1]

            acc_dis[i] = acc_dis[i-1] + mfv[i]
        df['acc_dis'] = acc_dis

        return df.acc_dis, df.mfv


    def obv(dataframe):
        """ on-balance-volume indicator """
        df = dataframe.copy()
        volume = df.volume
        price = df.close  
        obv = np.array(volume)

        for i in range(0, len(df)):  
            if i < 1:  
                continue  
            if price[i] > price[i-1]: new = volume[i]
            if price[i] == price[i-1]: new = 0.0
            if price[i] < price[i-1]: new = -volume[i]
            obv[i] = obv[i-1] + new

        df['obv'] = obv
        return df.obv


    # calculate simple moving averages of closing price
    df[f'SMA_{PERIOD}'] = df['close'].rolling(PERIOD).mean()
    df[f'SMA_{PERIOD_FAST}'] = df['close'].rolling(PERIOD_FAST).mean()

    # calculate simple moving averages of volume
    df[f'vol SMA_{PERIOD}'] = df['volume USD'].rolling(PERIOD).mean()
    df[f'vol SMA_{PERIOD_FAST}'] = df['volume USD'].rolling(PERIOD_FAST).mean()

    # centralise price and volume data around relevant slower SMA
    df['open_'] = (df['open'] - df[f'SMA_{PERIOD}']) / df[f'SMA_{PERIOD}']
    df['high_'] = (df['high'] - df[f'SMA_{PERIOD}']) / df[f'SMA_{PERIOD}']
    df['low_'] = (df['low'] - df[f'SMA_{PERIOD}']) / df[f'SMA_{PERIOD}']
    df['close_'] = (df['close'] - df[f'SMA_{PERIOD}']) / df[f'SMA_{PERIOD}']
    df['vol_'] = (df['volume USD'] - df[f'vol SMA_{PERIOD}']) / df[f'vol SMA_{PERIOD}']

    # centralise price and volume data around relevant faster SMA
    df['close_f'] = (df['close'] - df[f'SMA_{PERIOD_FAST}']) / df[f'SMA_{PERIOD_FAST}']
    df['vol_f'] = (df['volume USD'] - df[f'vol SMA_{PERIOD_FAST}']) / df[f'vol SMA_{PERIOD_FAST}']

    # calculate highest high and lowest low in the last 'PERIOD_FAST*2' prices
    df[f'HH{PERIOD_FAST*2}'] = df['high'].rolling(PERIOD_FAST*2).max().shift()
    df[f'LL{PERIOD_FAST*2}'] = df['low'].rolling(PERIOD_FAST*2).min().shift()

    # close price fraction of highest high and lowest low 
    df['chh_'] = (df['close'] - df[f'HH{PERIOD_FAST*2}']) / df[f'HH{PERIOD_FAST*2}']
    df['cll_'] = (df[f'LL{PERIOD_FAST*2}'] - df['close']) / df[f'LL{PERIOD_FAST*2}']

    # calculate bollinger coeffient and standard deviation add as column to dataframe
    df['bk'], df['d_sdev'] = bollinger_k(df, length=PERIOD_FAST)

    # deltas
    df['c-c1'] = df['close'] - df['close'].shift(1)
    df['h-h1'] = df['high'] - df['high'].shift(1)
    df['l-l1'] = df['low'] - df['low'].shift(1)
    df['v-v1'] = df['volume'] - df['volume'].shift(1)
    df['hlc-hlc1'] = (df['high'] + df['low'] + df['close'] / 3) - (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1) / 3)
    df['vwp-vwp1'] = ((df['high'] + df['low'] + df['close'] / 3) * df['volume']) - ((df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1) / 3) * df['volume'].shift(1))

    # relative volume-weighted average price
    df['rr_wvap_fast'] = rr_vwap(df, length=7)
    df['rr_wvap'] = rr_vwap(df, length=PERIOD)

    # volume indicators
    df['obv'] = obv(df)
    df['acc_dis'], df['mfv'] = acc_dis_mfv(df)

    # ishimoku cloud relative to close price
    df['tenkan'] = ((df['high'].rolling(ISHIMOKU_SCALE * 1).max() + df['low'].rolling(ISHIMOKU_SCALE * 1).min()) / 2) / df['close']
    df['kijun'] = ((df['high'].rolling(ISHIMOKU_SCALE * 3).max() + df['low'].rolling(ISHIMOKU_SCALE * 3).min()) / 2) / df['close']
    df['senkou_a'] = (df['tenkan'] + df['kijun']) / 2
    df['senkou_b'] = (df['high'].rolling(ISHIMOKU_SCALE * 6).max() + df['low'].rolling(ISHIMOKU_SCALE * 6).min()) / 2
    df['senkou'] = (df['senkou_a'] - df['senkou_b']) / df['close']
    df['senkou_a'] = df['senkou_a'] / df['close']
    df['senkou_b'] = df['senkou_b'] / df['close']

    # fractal adaptive moving average
    df['d_frama'], df['rel_frama'], df['frac_dim'], df['d_frac_dim'] = frama(df, batch=10)

    # negative group delay bandpass array
    df['ngd_3'] = z_transformer(df, mode='bandpass', period=3)[0]   
    df['ngd_4'] = z_transformer(df, mode='bandpass', period=4)[0]   
    df['ngd_6'] = z_transformer(df, mode='bandpass', period=6)[0]      
    df['ngd_8'] = z_transformer(df, mode='bandpass', period=8)[0]     
    df['ngd_11'] = z_transformer(df, mode='bandpass', period=11)[0]    
    df['ngd_16'] = z_transformer(df, mode='bandpass', period=16)[0]    
    df['ngd_22'] = z_transformer(df, mode='bandpass', period=22)[0]   
    df['ngd_32'] = z_transformer(df, mode='bandpass', period=32)[0]   

    # detrended relative strength index
    df['drsi_7'] = detrended_rsi(df, hp_threshold=7, rsi_period=7)
    df['drsi_14'] = detrended_rsi(df, hp_threshold=14, rsi_period=14)
    df['drsi_21'] = detrended_rsi(df, hp_threshold=21, rsi_period=14)

    # # capture indices of NaN rows to also remove from outcomes list
    # nan_indices = df.index[df.isnull().any(axis=1)].tolist()

    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)


    ## Calculate profit/loss fraction for each trade direction and timestamp

    up_outcomes = []
    up_outcomes_passive = []

    # tp_frac = ((df.loc[i, 'close'] + (ATR_MULT * df.loc[i, 'ATR'])) / df.loc[i, 'close']) - COMMISSION_FRAC
    # sl_frac = ((df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']) / df.loc[i, 'close']) - COMMISSION_FRAC

    for i in range(len(df)-2):

        if bool(random.getrandbits(1)) == True:    # randomises order of whether tp or sl is hit first if both are crossed

            if df.loc[i+1, 'low'] < (df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']):
                up_outcomes.append(((df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']) / df.loc[i, 'close']) - COMMISSION_FRAC)  

            elif df.loc[i+1, 'high'] > (df.loc[i, 'close'] + ATR_MULT * df.loc[i, 'ATR']):    
                up_outcomes.append(((df.loc[i, 'close'] + (ATR_MULT * df.loc[i, 'ATR'])) / df.loc[i, 'close']) - COMMISSION_FRAC)

            elif df.loc[i+2, 'low'] < (df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']):
                up_outcomes.append(((df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']) / df.loc[i, 'close']) - COMMISSION_FRAC)

            elif df.loc[i+2, 'high'] > (df.loc[i, 'close'] + ATR_MULT * df.loc[i, 'ATR']):
                up_outcomes.append(((df.loc[i, 'close'] + (ATR_MULT * df.loc[i, 'ATR'])) / df.loc[i, 'close']) - COMMISSION_FRAC)

            else:
                up_outcomes.append((df.loc[i+2, 'close'] / df.loc[i, 'close']) - COMMISSION_FRAC)

            up_outcomes_passive.append((df.loc[i+2, 'close'] / df.loc[i, 'close']) - (1.5 * COMMISSION_FRAC))

        else:
            if df.loc[i+1, 'high'] > (df.loc[i, 'close'] + ATR_MULT * df.loc[i, 'ATR']):    
                up_outcomes.append(((df.loc[i, 'close'] + (ATR_MULT * df.loc[i, 'ATR'])) / df.loc[i, 'close']) - COMMISSION_FRAC)      

            elif df.loc[i+1, 'low'] < (df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']):
                up_outcomes.append(((df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']) / df.loc[i, 'close']) - COMMISSION_FRAC)  

            elif df.loc[i+2, 'high'] > (df.loc[i, 'close'] + ATR_MULT * df.loc[i, 'ATR']):
                up_outcomes.append(((df.loc[i, 'close'] + (ATR_MULT * df.loc[i, 'ATR'])) / df.loc[i, 'close']) - COMMISSION_FRAC)

            elif df.loc[i+2, 'low'] < (df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']):
                up_outcomes.append(((df.loc[i, 'low'] - ATR_MULT * df.loc[i, 'ATR']) / df.loc[i, 'close']) - COMMISSION_FRAC)

            else:
                up_outcomes.append((df.loc[i+2, 'close'] / df.loc[i, 'close']) - COMMISSION_FRAC)

            up_outcomes_passive.append((df.loc[i+2, 'close'] / df.loc[i, 'close']) - (1.5 * COMMISSION_FRAC))

    up_outcomes.extend([float('NaN'), float('NaN')])
    up_outcomes_passive.extend([float('NaN'), float('NaN')])


    dn_outcomes = []
    dn_outcomes_passive = []

    # tp_frac = (df.loc[i, 'close'] / (df.loc[i, 'close'] - (ATR_MULT * df.loc[i, 'ATR']))) - COMMISSION_FRAC 
    # sl_frac = (df.loc[i, 'close'] / (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR'])) - COMMISSION_FRAC   

    for i in range(len(df)-2):

        if bool(random.getrandbits(1)) == True:    

            if df.loc[i+1, 'high'] > (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR']):
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR'])) - COMMISSION_FRAC) 

            elif df.loc[i+1, 'low'] < (df.loc[i, 'close'] - ATR_MULT * df.loc[i, 'ATR']):    
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'close'] - (ATR_MULT * df.loc[i, 'ATR']))) - COMMISSION_FRAC) 

            elif df.loc[i+2, 'high'] > (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR']):
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR'])) - COMMISSION_FRAC) 

            elif df.loc[i+2, 'low'] < (df.loc[i, 'close'] - ATR_MULT * df.loc[i, 'ATR']):
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'close'] - (ATR_MULT * df.loc[i, 'ATR']))) - COMMISSION_FRAC)

            else:
                dn_outcomes.append((df.loc[i, 'close'] / df.loc[i+2, 'close']) - COMMISSION_FRAC)

            dn_outcomes_passive.append((df.loc[i, 'close'] / df.loc[i+2, 'close']) - (1.5 * COMMISSION_FRAC))

        else:    

            if df.loc[i+1, 'low'] < (df.loc[i, 'close'] - ATR_MULT * df.loc[i, 'ATR']):    
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'close'] - (ATR_MULT * df.loc[i, 'ATR']))) - COMMISSION_FRAC) 

            elif df.loc[i+1, 'high'] > (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR']):
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR'])) - COMMISSION_FRAC) 

            elif df.loc[i+2, 'low'] < (df.loc[i, 'close'] - ATR_MULT * df.loc[i, 'ATR']):
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'close'] - (ATR_MULT * df.loc[i, 'ATR']))) - COMMISSION_FRAC)

            elif df.loc[i+2, 'high'] > (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR']):
                dn_outcomes.append((df.loc[i, 'close'] / (df.loc[i, 'high'] + ATR_MULT * df.loc[i, 'ATR'])) - COMMISSION_FRAC) 

            else:
                dn_outcomes.append((df.loc[i, 'close'] / df.loc[i+2, 'close']) - COMMISSION_FRAC)

            dn_outcomes_passive.append((df.loc[i, 'close'] / df.loc[i+2, 'close']) - (1.5 * COMMISSION_FRAC))

    dn_outcomes.extend([float('NaN'), float('NaN')])
    dn_outcomes_passive.extend([float('NaN'), float('NaN')])

    # filter results to match slicing of input tensors
    up_outcomes = [up_outcomes[i] for i in range(WINDOW, len(df), STEP)]
    dn_outcomes = [dn_outcomes[i] for i in range(WINDOW, len(df), STEP)]
    up_outcomes_passive = [up_outcomes_passive[i] for i in range(WINDOW, len(df), STEP)]
    dn_outcomes_passive = [dn_outcomes_passive[i] for i in range(WINDOW, len(df), STEP)]


    ## Separate catagorical dataframes and convert to one-hot encoding.

    # create separate dataframe for one-hot encoded day/hour categories
    df_time = df[['day', 'hour', 'year']].copy()
    df_time = pd.get_dummies(df_time, columns=['day', 'hour', 'year'])

    # create separate dataframe for one-hot encoded target categories
    df_ylabels = df['target'].copy()
    df_ylabels = pd.get_dummies(df_ylabels, columns=['target'])

    # tidy up price data
    df = df.drop(columns=['unix', 'date', 'volume USD', 'volume', 
                        'open', 'high', 'low', 'close', 
                        f'SMA_{PERIOD}', f'SMA_{PERIOD_FAST}', 
                        f'vol SMA_{PERIOD}', f'vol SMA_{PERIOD_FAST}', 
                        'day', 'hour', 'year', 'target', 'ATR', 'open_',
                        f'HH{PERIOD_FAST*2}', f'LL{PERIOD_FAST*2}', 
                        ], axis=1)


    ## Scale standard deviations of selected columns then check overall balance of data

    if PERCENT_CHANGE == True:
        df = df.pct_change()
        df = df.dropna(axis=0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # scale column stdev with stdev(close_)
    df['high_'] = df['high_'] / df['close_'].std()
    df['low_'] = df['low_'] / df['close_'].std()
    df['close_'] = df['close_'] / df['close_'].std()

    # scale column stdev to unity
    df['vol_'] = (df['vol_'] - df['vol_'].mean()) / df['vol_'].std() 
    df['close_f'] = df['close_f'] / df['close_f'].std()
    df['vol_f'] = df['vol_f'] / df['vol_f'].std()
    df['chh_'] = (df['chh_'] - df['chh_'].mean()) / df['chh_'].std()
    df['cll_'] = (df['cll_'] - df['cll_'].mean()) / df['cll_'].std()
    df['bk'] = df['bk'] / df['bk'].std()
    df['d_sdev'] = df['d_sdev'] / df['d_sdev'].std()
    df['rr_wvap_fast'] = (df['rr_wvap_fast'] - df['rr_wvap_fast'].mean()) / df['rr_wvap_fast'].std()
    df['rr_wvap'] = (df['rr_wvap'] - df['rr_wvap'].mean()) / df['rr_wvap'].std()
    df['vwp-vwp1'] = df['vwp-vwp1'] / df['vwp-vwp1'].std()

    df['c-c1'] = df['c-c1'] / df['c-c1'].std()
    df['h-h1'] = df['h-h1'] / df['h-h1'].std()      
    df['l-l1'] = df['l-l1'] / df['l-l1'].std()
    df['v-v1'] = df['v-v1'] / df['v-v1'].std()
    df['hlc-hlc1'] = df['hlc-hlc1'] / df['hlc-hlc1'].std()

    df['tenkan'] = (df['tenkan'] - df['tenkan'].mean()) / df['tenkan'].std()    
    df['kijun'] = (df['kijun'] - df['kijun'].mean()) / df['kijun'].std()
    df['senkou'] = (df['senkou'] - df['senkou'].mean()) / df['senkou'].std()
    df['senkou_a'] = (df['senkou_a'] - df['senkou_a'].mean()) / df['senkou_a'].std()
    df['senkou_b'] = (df['senkou_b'] - df['senkou_b'].mean()) / df['senkou_b'].std()

    df['acc_dis'] = (df['acc_dis'] - df['acc_dis'].mean()) / df['acc_dis'].std()
    df['obv'] = (df['obv'] - df['obv'].mean()) / df['obv'].std()
    df['mfv'] = (df['mfv'] - df['mfv'].mean()) / df['mfv'].std()

    df['d_frama'] = (df['d_frama'] - df['d_frama'].mean()) / df['d_frama'].std()    
    df['rel_frama'] = (df['rel_frama'] - df['rel_frama'].mean()) / df['rel_frama'].std()
    df['frac_dim'] = (df['frac_dim'] - df['frac_dim'].mean()) / df['frac_dim'].std()
    df['d_frac_dim'] = (df['d_frac_dim'] - df['d_frac_dim'].mean()) / df['d_frac_dim'].std()

    df['drsi_7'] = (df['drsi_7'] - df['drsi_7'].mean()) / df['drsi_7'].std()
    df['drsi_14'] = (df['drsi_14'] - df['drsi_14'].mean()) / df['drsi_14'].std()
    df['drsi_21'] = (df['drsi_21'] - df['drsi_21'].mean()) / df['drsi_21'].std()

    df['ngd_3'] = (df['ngd_3'] - df['ngd_3'].mean()) / df['ngd_3'].std()
    df['ngd_4'] = (df['ngd_4'] - df['ngd_4'].mean()) / df['ngd_4'].std()
    df['ngd_6'] = (df['ngd_6'] - df['ngd_6'].mean()) / df['ngd_6'].std()
    df['ngd_8'] = (df['ngd_8'] - df['ngd_8'].mean()) / df['ngd_8'].std()
    df['ngd_11'] = (df['ngd_11'] - df['ngd_11'].mean()) / df['ngd_11'].std()
    df['ngd_16'] = (df['ngd_16'] - df['ngd_16'].mean()) / df['ngd_16'].std()
    df['ngd_22'] = (df['ngd_22'] - df['ngd_22'].mean()) / df['ngd_22'].std()
    df['ngd_32'] = (df['ngd_32'] - df['ngd_32'].mean()) / df['ngd_32'].std()


    ## Prepare Tensors

    # @jit
    def prepare_tensors(df, df_time, df_ylabels, window, step): 

        # create numpy arrays to receive data
        price_series = np.zeros(shape=(window, len(df.columns)))
        time_categories = np.zeros(shape=(1, len(df_time.columns)))
        target_categories = np.zeros(shape=(1, 3))

        # iterate through price dataframe concatenating discrete arrays of size 'window', and spacing 'step'
        batch_size = (len(df)-window) // step
        for i in range(batch_size): 
            arr = df.iloc[[(i*step)+j for j in range(window)], [k for k in range(len(df.columns))]].to_numpy()
            price_series = np.concatenate((price_series, arr))

        # iterate through categorical dataframes concatenating data relating to bottom row of each price window
        for i in range(batch_size): 
            arr = df_time.iloc[[(i*step) + window], : ].to_numpy()
            time_categories = np.concatenate((time_categories, arr))
            
            arr = df_ylabels.iloc[[(i*step) + window], : ].to_numpy()
            target_categories = np.concatenate((target_categories, arr))

        # reshape arrays
        price_series = np.reshape(price_series, (batch_size+1, window, len(df.columns)))
        time_categories = np.reshape(time_categories, (batch_size+1, len(df_time.columns)))
        target_categories = np.reshape(target_categories, (batch_size+1, 3))

        # delete intial 'zeros' array elements
        price_series = np.delete(price_series, 0, axis=0)
        time_categories = np.delete(time_categories, 0, axis=0)
        target_categories = np.delete(target_categories, 0, axis=0)

        # binarize target categories, index(1) == positive class
        up_targets = target_categories[ : , [1, 0]]
        dn_targets = target_categories[ : , [1, 2]]

        for element in up_targets:
            if element[0] == 0 and element[1] == 0:
                element[0] = 1

        for element in dn_targets:
            if element[0] == 0 and element[1] == 0:
                element[0] = 1

        return price_series, time_categories, up_targets, dn_targets


    price_series, time_categories, up_targets, dn_targets = prepare_tensors(df, df_time, df_ylabels, WINDOW, STEP)

    timesteps = price_series.shape[1]
    channels = price_series.shape[2]
    time_dim = time_categories.shape[1]


    ## Split Data

    TARGETS = dn_targets
    OUTCOMES = dn_outcomes

    # Splitting the arrays into train and test sets
    train_x_price, test_x_price = train_test_split(price_series, test_size=0.2, random_state=RANDOM_SEED)
    train_x_time, test_x_time = train_test_split(time_categories, test_size=0.2, random_state=RANDOM_SEED)
    train_y, test_y = train_test_split(TARGETS, test_size=0.2, random_state=RANDOM_SEED)
    train_outcomes, test_outcomes = train_test_split(OUTCOMES, test_size=0.2, random_state=RANDOM_SEED)


    ## Define Multimodal Convolutional Network

    keras.utils.set_random_seed(7)

    # convolutional branch
    input_cnn = Input(shape=(timesteps,channels)) 

    noise_cnn = GaussianNoise(stddev=NOISE_SDEV, seed=7)(input_cnn)

    cnn = Conv1D(filters=FILTERS, kernel_size=7, padding='same', 
                activation='relu', data_format='channels_last',
                activity_regularizer=None,
                input_shape=(timesteps, channels))(noise_cnn)

    cnn = Conv1D(filters=FILTERS, kernel_size=3, padding='same', 
                activation='relu', data_format='channels_last', 
                activity_regularizer=None)(cnn)

    cnn = Conv1D(filters=FILTERS, kernel_size=2, padding='same', strides=2,      # pooling layer
                activation='relu', use_bias=False, data_format='channels_last', 
                activity_regularizer=None)(cnn)

    cnn = Conv1D(filters=FILTERS*2, kernel_size=3, padding='same', 
                activation='relu', data_format='channels_last', 
                activity_regularizer=None)(cnn)

    cnn = Conv1D(filters=FILTERS*2, kernel_size=2, padding='same', strides=2,     # pooling layer
                activation='relu', use_bias=False, data_format='channels_last',
                activity_regularizer=None)(cnn)

    cnn = Conv1D(filters=FILTERS*4, kernel_size=3, padding='same', 
                activation='relu', data_format='channels_last', 
                activity_regularizer=regularizers.L2(0.01))(cnn)

    cnn = Dropout(0.4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Model(inputs=input_cnn, outputs=cnn)

    # perceptron branch
    input_mlp = Input(shape=(time_dim,))
    noise_mlp = GaussianNoise(stddev=DEEP_NOISE_SDEV, seed=7)(input_mlp)
    mlp = Dense(8, activation='relu', 
                activity_regularizer=regularizers.L2(0.01))(noise_mlp)
    mlp = Model(inputs=input_mlp, outputs=mlp)

    # join branches
    combined = concatenate([cnn.output, mlp.output])
    head = Dense(1024, activation='relu')(combined)
    head = Dense(512, activation='relu')(head)
    head = Dense(2, activation='softmax')(head)
    model = Model(inputs=[cnn.input, mlp.input], outputs=head)

    loss = keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False,    # ideally matches metric, but: https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
                                                alpha=0.25,
                                                gamma=2.0)    
    
    metric = keras.metrics.F1Score(average='weighted',    
                                   threshold=None, 
                                   dtype=None)     
                                                            
    model.compile(loss=loss,     
                  optimizer='adam',     # adadelta or adamax may be better suited?                
                  metrics=[metric])     # does list block optuna?


    ## Train And Evaluate Model

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",  
   				 					 	           patience=PATIENCE,
    			 					  	           verbose=0,
				 						           restore_best_weights=True,)

    optuna_pruning = KerasPruningCallback(trial, "val_loss")

    postive_weight = train_y.sum(axis=0)[0] / train_y.sum(axis=0)[1]

    model.fit(x=[train_x_price, train_x_time], 
              y=train_y,
              validation_data=([test_x_price, test_x_time], test_y),
              epochs=25, 
              batch_size=BATCH_SIZE, 
              callbacks=[early_stopping, optuna_pruning],
              class_weight = {0: 1,
                              1: postive_weight})

    loss, f1_score = model.evaluate([test_x_price, test_x_time], 
                                    test_y, 
                                    batch_size=BATCH_SIZE, 
                                    verbose=0)
    print(loss, f1_score)

    return loss


if __name__ == "__main__":

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Visualize the optimization history.
    plot_optimization_history(study).show()

    # Visualize the learning curves of the trials.
    plot_intermediate_values(study).show()

    # Visualize high-dimensional parameter relationships.
    plot_parallel_coordinate(study).show()

    # Select parameters to visualize.
    plot_parallel_coordinate(study, params=["lr_init", "n_units_l0"]).show()

    # Visualize hyperparameter relationships.
    plot_contour(study).show()

    # Select parameters to visualize.
    plot_contour(study, params=["n_units_l0", "n_units_l1"]).show()

    # Visualize individual hyperparameters.
    plot_slice(study).show()

    # Select parameters to visualize.
    plot_slice(study, params=["n_units_l0", "n_units_l1"]).show()

    # Visualize parameter importances.
    plot_param_importances(study).show()