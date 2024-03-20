""" converts open-high-low-close price data into a format suitable for machine learning.
    Target labels are created according to profitable trading conditions being met.  
    Price and volume data are normalised around moving averages, whilst time and day 
    labels are separated out and one-hot encoded. """


import pandas as pd
import time


def convert_unix_time_into_day(seconds):
    """ returns day of week as category label """
    day_of_week = time.strftime('%A', time.localtime(seconds))
    dict = {'Saturday':1, 'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday':5, 'Thursday':6, 'Friday':7}
    day_code = dict[day_of_week]
    return day_code


def wwma(values, n):
    """ w. wilder's EMA """
    return values.ewm(alpha=1/n, adjust=False, ignore_na=True).mean()


def atr(df, length):
    """ average true range """
    high, low, prev_close = df['high'], df['low'], df['close'].shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)
    atr = wwma(tr, length)
    return atr


SMA_PERIOD = 50
SMA_PERIOD_FAST = 20
SPREAD = 10   # replace with ATR-based calculation


# load price data into dataframe
df = pd.read_csv('OHLC_1h.csv', header=0, sep=',')

# create target labels 
y_labels =[float('NaN'), float('NaN')]

for i in range(len(df)-2):

    # up condition: (high or high[1] > high[2] + spread) and (low and low[1] > low[2] + spread)
    if (
        ( df.loc[i, 'high'] > (df.loc[i+2, 'high'] + SPREAD)  or  df.loc[i+1, 'high'] > (df.loc[i+2, 'high'] + SPREAD) )  and 
        ( df.loc[i, 'low'] > (df.loc[i+2, 'low'] + SPREAD)  and  df.loc[i+1, 'low'] > (df.loc[i+2, 'low'] + SPREAD) )
    ):
        label = 'up'

    # down condition: (low or low[1] < low[2] - spread) and (high and high[1] < high[2] - spread)
    elif (
        ( df.loc[i, 'low'] < (df.loc[i+2, 'low'] - SPREAD)  or  df.loc[i+1, 'low'] < (df.loc[i+2, 'low'] - SPREAD) )  and 
        ( df.loc[i, 'high'] < (df.loc[i+2, 'high'] - SPREAD)  and  df.loc[i+1, 'high'] < (df.loc[i+2, 'high'] - SPREAD) )
        ):
        label = 'down'
    
    else:
        label = 'flat'

    y_labels.append(label)

# add labels to new dataframe column
df['target'] = y_labels

# calculate simple moving averages of closing price from bottom upwards
df[f'SMA_{SMA_PERIOD}'] = df['close'].rolling(SMA_PERIOD).mean().shift(-SMA_PERIOD +1)
df[f'SMA_{SMA_PERIOD_FAST}'] = df['close'].rolling(SMA_PERIOD_FAST).mean().shift(-SMA_PERIOD_FAST +1)

# calculate simple moving averages of volume from bottom upwards
df[f'vol SMA_{SMA_PERIOD}'] = df['volume USD'].rolling(SMA_PERIOD).mean().shift(-SMA_PERIOD +1)
df[f'vol SMA_{SMA_PERIOD_FAST}'] = df['volume USD'].rolling(SMA_PERIOD_FAST).mean().shift(-SMA_PERIOD_FAST +1)

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
                      'SMA_20', 'vol SMA_20', 'day', 'hour', 'target'], axis=1)

# check dataframes
print(df.head(), '\n', len(df))
print(df_time.head(), '\n', len(df_time))
print(df_ylabels.head(), '\n', len(df_ylabels))


## do negative numbers affect convolution?
## standardise
## fix ATR
