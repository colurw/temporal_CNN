import pandas as pd
import time


def convert_unix_time_into_day(seconds):
    day_of_week = time.strftime('%A', time.localtime(seconds))
    dict = {'Saturday':1, 'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday':5, 'Thursday':6, 'Friday':7}
    day_code = dict[day_of_week]
    return day_code


SMA_PERIOD = 50
SMA_PERIOD_FAST = 20

df = pd.read_csv('OHLC_1h.csv', header=0, sep=',')

# calculate simple moving averages of closing price from bottom upwards
df[f'SMA_{SMA_PERIOD}'] = df['close'].rolling(SMA_PERIOD).mean().shift(-SMA_PERIOD +1)
df[f'SMA_{SMA_PERIOD_FAST}'] = df['close'].rolling(SMA_PERIOD_FAST).mean().shift(-SMA_PERIOD_FAST +1)

# calculate simple moving averages of volume from bottom upwards
df[f'vol SMA_{SMA_PERIOD}'] = df['volume USD'].rolling(SMA_PERIOD).mean().shift(-SMA_PERIOD +1)
df[f'vol SMA_{SMA_PERIOD_FAST}'] = df['volume USD'].rolling(SMA_PERIOD_FAST).mean().shift(-SMA_PERIOD_FAST +1)
df = df.dropna(axis=0)

# normalise data as fractional difference from relevant SMA
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

# create separate one-hot encoded dataframe using day and hour categories
df_time = df[['day', 'hour']].copy()
df_time = pd.get_dummies(df_time, columns=['day', 'hour'])

# tidy up
df = df.drop(columns=['unix', 'date', 'volume USD', 'volume BTC', 'symbol', 
                      'open', 'high', 'low', 'close', 'SMA_50', 'vol SMA_50', 
                      'SMA_20', 'vol SMA_20', 'day', 'hour'], axis=1)

# check
print(df_time.head(30))
print(df.head(30))


## do negative numbers affect convolution?
## convert daycode to onehot
## consider adding second moving average
