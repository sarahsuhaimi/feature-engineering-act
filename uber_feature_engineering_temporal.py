# %%
import datetime
import numpy as np
import pandas as pd
import pytz

# %%
df_cab = pd.read_csv('rideshare_kaggle.csv')
df_cab.head()

# %%
# CHECKING MISSING DATA
missing = df_cab.isnull().sum().sum()
print(f'Total missing data: {missing}')
print(df_cab.isnull().sum())
# DO NOTHING BECAUSE WE ONLY DEAL WITH DATETIME COLUMN

# %%
date_time = df_cab['datetime']
date_time.head()

# %%
df = pd.DataFrame(date_time, columns=['datetime'])
df

# %%
# CONVERT NORMAL SERIES INTO A TIME SERIES ARRAY
ts = np.array([pd.Timestamp(item) for item in np.array(df_cab['datetime'])])
df['ts_obj'] = ts

# %%
df

# %%
df['year'] = df['ts_obj'].apply(lambda d: d.year)
df['month'] = df['ts_obj'].apply(lambda d: d.month)
df['day'] = df['ts_obj'].apply(lambda d: d.day)
df['day_of_week'] = df['ts_obj'].apply(lambda d: d.dayofweek)
df['quarter'] = df['ts_obj'].apply(lambda d: d.quarter)

df[['datetime', 'year', 'month', 'day', 'day_of_week', 'quarter']]

# %%
df['hour'] = df['ts_obj'].apply(lambda d: d.hour)
df['minute'] = df['ts_obj'].apply(lambda d: d.minute)
df['second'] = df['ts_obj'].apply(lambda d: d.second)

df[['datetime', 'hour', 'minute', 'second']]

# %%
# BINNING THE HOUR FEATURE
# CAN GET INFO ON BILA UBER IN DEMAND
hour_bins = [-1, 5, 11, 16, 21, 23]
bin_names = ['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']

df['time_of_day_bin'] = pd.cut(df['hour'],
                               bins=hour_bins, labels=bin_names)

df[['datetime','hour','time_of_day_bin']]

# %%



