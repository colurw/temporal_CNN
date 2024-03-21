""" converts open-high-low-close price data into a format suitable for machine learning.
    Target labels are created according to profitable trading conditions being met.  
    Price and volume data are normalised around moving averages, whilst time and day 
    labels are separated out and one-hot encoded. 
    Dataframe is then transformed into an array of dimensions [batch_size, steps, channels] """


import pandas as pd
import numpy as np
import time


def convert_unix_time_into_day(seconds):
    """ returns day of week as category label """
    day_of_week = time.strftime('%A', time.localtime(seconds))
    dict = {'Saturday':1, 'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday':5, 'Thursday':6, 'Friday':7}
    day_code = dict[day_of_week]
    return day_code


def wwma(pd_series, period):
    """ w. wilder's EMA """
    return pd_series.ewm(alpha=1/period, adjust=False, ignore_na=True).mean()


def atr(df, length=14):
    """ average true range (for column with latest values at top) """
    df_high, df_low, df_prev_close = df['high'], df['low'], df['close'].shift()
    df_tr = [df_high- df_low, df_high - df_prev_close, df_low - df_prev_close]
    df_tr = [tr.abs() for tr in df_tr]
    df_tr = pd.concat(df_tr, axis=1).max(axis=1)
    df_atr = wwma(df_tr, length)
    return df_atr


# feature engineering constants
SMA_PERIOD = 50
SMA_PERIOD_FAST = 20
ATR_MULT = 0.25 

# data preparation constants
CHANNELS = 10
WINDOW = 5
STEP = 2


# load price data into dataframe and reorder to recent = last
df = pd.read_csv('OHLC_1h.csv', header=0, sep=',')
df = df.sort_index(ascending=False, ignore_index=True)

# calculate average true range and add as column to dataframe
df_atr = atr(df)
df['ATR'] = df_atr

# create target labels 
y_labels =[]

for i in range(2,len(df)):

    # up condition: (high or high[1] > high[2] + spread) and (low and low[1] > low[2] + spread) 
    if (
        ( df.loc[i, 'high'] > (df.loc[i-2, 'high'] + ATR_MULT * df.loc[i, 'ATR'])  or  
        df.loc[i-1, 'high'] > (df.loc[i-2, 'high'] + ATR_MULT * df.loc[i, 'ATR']) )  and 
        ( df.loc[i, 'low'] > (df.loc[i-2, 'low'] + ATR_MULT * df.loc[i, 'ATR'])  and  
        df.loc[i-1, 'low'] > (df.loc[i-2, 'low'] + ATR_MULT * df.loc[i, 'ATR']) )
    ):
        label = 'up'

    # down condition: (low or low[1] < low[2] - spread) and (high and high[1] < high[2] - spread)
    elif (
        ( df.loc[i, 'low'] < (df.loc[i-2, 'low'] - ATR_MULT * df.loc[i, 'ATR'])  or  
          df.loc[i-1, 'low'] < (df.loc[i-2, 'low'] - ATR_MULT * df.loc[i, 'ATR']) )  and 
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

# calculate simple moving averages of closing price
df[f'SMA_{SMA_PERIOD}'] = df['close'].rolling(SMA_PERIOD).mean()
df[f'SMA_{SMA_PERIOD_FAST}'] = df['close'].rolling(SMA_PERIOD_FAST).mean()

# calculate simple moving averages of volume
df[f'vol SMA_{SMA_PERIOD}'] = df['volume USD'].rolling(SMA_PERIOD).mean()
df[f'vol SMA_{SMA_PERIOD_FAST}'] = df['volume USD'].rolling(SMA_PERIOD_FAST).mean()

# normalise data as fractional difference from relevant slower SMA
df['open_'] = (df['open'] - df[f'SMA_{SMA_PERIOD}']) / df[f'SMA_{SMA_PERIOD}']
df['high_'] = (df['high'] - df[f'SMA_{SMA_PERIOD}']) / df[f'SMA_{SMA_PERIOD}']
df['low_'] = (df['low'] - df[f'SMA_{SMA_PERIOD}']) / df[f'SMA_{SMA_PERIOD}']
df['close_'] = (df['close'] - df[f'SMA_{SMA_PERIOD}']) / df[f'SMA_{SMA_PERIOD}']
df['vol_'] = (df['volume USD'] - df[f'vol SMA_{SMA_PERIOD}']) / df[f'vol SMA_{SMA_PERIOD}']

# normalise data as fractional difference from relevant faster SMA
df['open_f'] = (df['open'] - df[f'SMA_{SMA_PERIOD_FAST}']) / df[f'SMA_{SMA_PERIOD_FAST}']
df['high_f'] = (df['high'] - df[f'SMA_{SMA_PERIOD_FAST}']) / df[f'SMA_{SMA_PERIOD_FAST}']
df['low_f'] = (df['low'] - df[f'SMA_{SMA_PERIOD_FAST}']) / df[f'SMA_{SMA_PERIOD_FAST}']
df['close_f'] = (df['close'] - df[f'SMA_{SMA_PERIOD_FAST}']) / df[f'SMA_{SMA_PERIOD_FAST}']
df['vol_f'] = (df['volume USD'] - df[f'vol SMA_{SMA_PERIOD_FAST}']) / df[f'vol SMA_{SMA_PERIOD_FAST}']

# create day and hour categories
df['day'] = df['unix'].map(lambda _: convert_unix_time_into_day(_))
df['hour'] = df['date'].str.slice(start=11, stop=13).apply(pd.to_numeric) + 1

# drop rows containing NaNs   
df = df.dropna(axis=0)

# create separate dataframe for one-hot encoded day/hour categories
df_time = df[['day', 'hour']].copy()
df_time = pd.get_dummies(df_time, columns=['day', 'hour'])

# create separate dataframe for one-hot encoded target categories
df_ylabels = df['target'].copy()
df_ylabels = pd.get_dummies(df_ylabels, columns=['target'])

# tidy up price data
df = df.drop(columns=['unix', 'date', 'volume USD', 'volume BTC', 'symbol', 
                      'open', 'high', 'low', 'close', 'SMA_50', 'vol SMA_50', 
                      'SMA_20', 'vol SMA_20', 'day', 'hour', 'target', 'ATR'], axis=1)

# check dataframes
print(df.head(), '\n', len(df))
print(df_time.head(), '\n', len(df_time))
print(df_ylabels.head(), '\n', len(df_ylabels))

# check data balance
print(df_ylabels.sum(axis=0))
print('mean:', df.stack().mean())


# create numpy arrays to receive data
price_series_data = np.zeros(shape=(WINDOW, CHANNELS))
time_cat_data = np.zeros(shape=(1, 31))
target_cat_data = np.zeros(shape=(1, 3))

batch_size = (len(df)-WINDOW) // STEP

# iterate through price dataframe concatenating discrete arrays of size 'WINDOW', and spacing 'STEP'
for i in range(batch_size): 
    arr = df.iloc[[(i*STEP)+j for j in range(WINDOW)], [k for k in range(CHANNELS)]].to_numpy()
    price_series_data = np.concatenate((price_series_data, arr))

# iterate through categorical dataframes concatenating data relating to bottom row of each price window
for i in range(batch_size): 

    arr = df_time.iloc[[(i*STEP) + WINDOW], : ].to_numpy()
    time_cat_data = np.concatenate((time_cat_data, arr))
    
    arr = df_ylabels.iloc[[(i*STEP) + WINDOW], : ].to_numpy()
    target_cat_data = np.concatenate((target_cat_data, arr))

# reshape arrays
price_series_data = np.reshape(price_series_data, (batch_size+1, WINDOW, CHANNELS))
time_cat_data = np.reshape(time_cat_data, (batch_size+1, 31))
target_cat_data = np.reshape(target_cat_data, (batch_size+1, 3))

# delete intial 'zeros' array elements
price_series_data = np.delete(price_series_data, 0, axis=0)
time_cat_data = np.delete(time_cat_data, 0, axis=0)
target_cat_data = np.delete(target_cat_data, 0, axis=0)

# check arrays
print(price_series_data, price_series_data.shape)
print(time_cat_data, time_cat_data.shape)
print(target_cat_data, target_cat_data.shape)


## match up labels and time categories with array[batch_size, steps, channels]
## do negative numbers affect convolution?
## standardise
## add commission level to 'flat' calculation
## add bollinger bands
## add highest_high(40) level

