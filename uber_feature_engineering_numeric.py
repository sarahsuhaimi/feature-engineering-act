# %%
# IMPORT PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as spstats

#get_ipython().magic('matplotlib inline')
mpl.style.reload_library()
mpl.style.use('classic')
mpl.rcParams['figure.facecolor'] = (1, 1, 1, 0)
mpl.rcParams['figure.figsize'] = [6.0, 4.0]
mpl.rcParams['figure.dpi'] = 100

# %%
# READ DATASET
df_cab = pd.read_csv('rideshare_kaggle.csv')

# %%
df_cab.head()

# %%
df_cab.dtypes

# %%
df_cab.shape

# %%
# Checking missing data
total_na = df_cab.isnull().sum().sum()
print(f'Total Missing Data: {total_na}')

# %%
# DROP THE MISSING DATA
df_clean_cab = df_cab.dropna(axis=0)

# %%
df_clean_cab.shape

# %%
df_clean_cab.head()

# %%
# ONLY PICK UBER DATA
df_clean_uber = df_clean_cab[df_clean_cab['cab_type'] == 'Uber']
df_clean_uber = df_clean_uber.reset_index(drop=True)
df_clean_uber.head()

# %%
dis_price = df_clean_uber[['distance', 'price']]
dis_price

# %%
# Interaction between distance and price
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
res = pf.fit_transform(dis_price)
res

# %%
pd.DataFrame(pf.powers_, columns=['distance', 'price'])

# %%
interact_features = pd.DataFrame(res, columns=['Distance','Price','Distance x Price'])
interact_features.head()

# %%
new_df = pd.DataFrame([[0.50, 5.50], [1.00, 10.00], [0.70, 6.50]],
                      columns=['distance', 'price'])
new_df

# %%
new_res = pf.transform(new_df)
new_intr_features = pd.DataFrame(new_res, 
                                 columns=['Distance','Price','Distance x Price'])
new_intr_features

# %%
# FIXED-WIDTH BINNING

fig, ax = plt.subplots()
df_clean_uber['temperature'].hist(color='violet')
ax.set_title('Uber Temperature Histogram', fontsize=12)
ax.set_xlabel('Temperature', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

# %%
# Temperature Range: Bin
# ----------------------
#       15  -   24 : 1
#       25  -   34 : 2
#       35  -   44 : 3
#       45  -   54 : 4

bin_ranges = [15, 24, 34, 44, 54]
bin_names = [1, 2, 3, 4]
df_clean_uber['temp_bin'] = pd.cut(np.array(df_clean_uber['temperature']), 
                                               bins=bin_ranges)
df_clean_uber['temp_bin_label'] = pd.cut(np.array(df_clean_uber['temperature']), 
                                               bins=bin_ranges, labels=bin_names)
df_clean_uber[['temperature', 'temp_bin', 
               'temp_bin_label']].iloc[10:15]


# %%
# ADAPTIVE BINNING ON UBER PRICE
quantile_list = [0, .25, .5, .75, 1]
quantiles = df_clean_uber['price'].quantile(quantile_list)
quantiles

# %%
fig, ax = plt.subplots()
df_clean_uber['price'].hist(bins=35, color='violet')
ax.set_title('Uber Price Histogram', fontsize=12)
ax.set_xlabel('Price', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

# %%
fig, ax = plt.subplots()
df_clean_uber['price'].hist(bins=35, color='violet')

for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)

ax.set_title('Uber Price Histogram with Quantiles', fontsize=12)
ax.set_xlabel('Uber Price', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

# %%
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
df_clean_uber['price_quantile_range'] = pd.qcut(df_clean_uber['price'], 
                                                 q=quantile_list)
df_clean_uber['price_quantile_label'] = pd.qcut(df_clean_uber['price'], 
                                                 q=quantile_list, labels=quantile_labels)
df_clean_uber.head()

# %%
# MATHEMATICAL TRANSFORMATION
# LOG TRANSFORM

df_clean_uber['price_log'] = np.log((1 + df_clean_uber['price']))
df_clean_uber[['price','price_log']].iloc[4:10]

# %%
price_log_mean = np.round(np.mean(df_clean_uber['price_log']), 2)

fig, ax = plt.subplots()
df_clean_uber['price_log'].hist(bins=35, color='violet')
plt.axvline(price_log_mean, color='r')
ax.set_title('Uber Price Histogram after Log Transform', fontsize=12)
ax.set_xlabel('Uber Price (log scale)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.text(11.5, 450, r'$\mu$='+str(price_log_mean), fontsize=10)

# %%
# BOX - COX TRANSFORM
price = np.array(df_clean_uber['price'])
price_clean = price[~np.isnan(price)]
l, opt_lambda = spstats.boxcox(price_clean)
print('Optimal lambda value: ', opt_lambda)

# %%
df_clean_uber['price_boxcox_lambda_0'] = spstats.boxcox((1+df_clean_uber['price']), 
                                                         lmbda=0)
df_clean_uber['price_boxcox_lambda_opt'] = spstats.boxcox(df_clean_uber['price'], 
                                                           lmbda=opt_lambda)
df_clean_uber[['price', 'price_log', 'price_boxcox_lambda_0', 'price_boxcox_lambda_opt']].iloc[4:10]

# %%
price_boxcox_mean = np.round(np.mean(df_clean_uber['price_boxcox_lambda_opt']), 2)

fig, ax = plt.subplots()
df_clean_uber['price_boxcox_lambda_opt'].hist(bins=35, color='violet')
plt.axvline(price_boxcox_mean, color='r')
ax.set_title('Uber Price Histogram after Box–Cox Transform', fontsize=12)
ax.set_xlabel('Uber Price (Box–Cox transform)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.text(24, 450, r'$\mu$='+str(price_boxcox_mean), fontsize=10)


