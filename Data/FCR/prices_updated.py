import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
#%% DATAFRAME CREATION

#IMPORT DATA
raw_data  = pd.read_csv('market_data/fcrreservesdk2_28_01_2022.csv')
df      = raw_data[[raw_data.columns[i] for i in [1, 3, 5]]].copy()

# INDEX SORTING TO DATETIME
df.set_index('HourDK', inplace = True)
df.index   = pd.to_datetime(df.index)
df  = df.sort_index(axis=0, ascending=True)

# DROP MISSING VALUES
df.dropna(inplace=True)

# PLOT TIME SERIES

linewidth = 0.5

fig, ax = plt.subplots(2, 1, sharex = True, sharey=True, figsize = (10, 5))
ax[0].plot(df['FCR_N_PriceEUR'], linewidth = linewidth, label = 'FCR-N ')
ax[0].set_ylabel('Capacity payment [€/MW]')
ax[0].legend()

ax[1].plot(df['FCR_D_UpPriceEUR'], linewidth = linewidth, label = 'FCR-D Up ', color = 'tab:orange')
ax[1].set_ylabel('Capacity payment [€/MW]')
ax[1].legend()
plt.tight_layout()
plt.savefig('images/FCR.png', dpi = 1000)
plt.show()
#%% SEASONAL DECOMPOSITION

# DATA SAMPLING
temp = df[(df.index.year == 2019) & (df.index.month == 1)]
fig, ax = plt.subplots(1, 1, sharex = True, sharey=True, figsize = (10, 5))
plt.step(temp.index[:], temp.iloc[:, 1])
plt.xticks(rotation = 'vertical')
plt.show()
# SEASONAL DECOMPOSITION
result=seasonal_decompose(temp['FCR_N_PriceEUR'], model='additive', period=24)

fig, ax = plt.subplots(3, 1, figsize = (10,5), sharex = True)
ax[0].set_title('Seasonality analysis through additive decomposition')
ax[0].plot(temp['FCR_N_PriceEUR'])
ax[0].set_ylabel('FCR-N price [€/MW]')
ax[1].plot(result.trend)
ax[1].set_ylabel('Trend')
ax[2].plot(result.seasonal)
ax[2].set_ylabel('Seasonality')
plt.xticks(rotation = 30)
plt.tight_layout()
plt.savefig('images/seasonality.png', dpi = 1000)
