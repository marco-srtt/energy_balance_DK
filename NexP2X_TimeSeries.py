import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import pylab
from datetime import datetime
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
#%%
# IMPORTING DATABASE

filename        = 'wind_pow.csv'
raw_csv_data    = pd.read_csv(filename)

# CREATING A COPY
df_comp         = raw_csv_data.copy()
AttributeNames  = list(df_comp.columns.values)
# GENERAL INFO 
#df_comp.head()        # if needed to check head of the dataset
#df_comp.describe()    # summary statistics of dataset
#df_comp.isna()        # check missing elements
#df_comp.isna().sum()  # number of available entries for each column
   
#%%

# DATA PREPARATION
#df_comp.Time = pd.to_datetime(df_comp.Time, dayfirst = True)    # Assumes date mm/dd/yyyy
for i in range(len(df_comp)):
    df_comp['Time'].values[i] = datetime.strptime(df_comp['Time'].values[i], '%d/%m/%Y %H.%S')

# INDEX

df_comp.set_index('Time', inplace = True)   # inplace create an integer indexes

# FREQUENCY
df_comp = df_comp.asfreq('h')               # set frequency of data

#%%
# PLOTTING

df_comp.DKe_wind.plot(figsize = (20,5))
df_comp.SE2_wind.plot(figsize = (20,5))
df_comp.SE4_wind.plot(figsize = (20,5))
plt.title('Preliminary analysis')
plt.show()

df_comp.plot(subplots=True)
plt.show()

# QQ PLOT
# showcases how data fits a Normal Distribution
# red line should be followed if normal distribution


scipy.stats.probplot(df_comp.DKe_wind, plot = pylab) 
pylab.show()
# MACHINE LEARNING
# MACHINE LEARNING

size = int(len(df_comp)*0.8)                # cannot shuffle, chronological order matters
df = df_comp.iloc[:size]                    # from start to size
df_test = df_comp.iloc[size:]               # from size to end

#%%
# WHITE NOISE                               # Time series data does not have a pattern
                                            # Contant mean, var and no autocorrelation
wn = np.random.normal(loc = df.DKe_wind.mean(), scale = df.DKe_wind.std(), size = len(df))  
df['wn'] = wn                                         
df.wn.plot(figsize=(20,5))
plt.title('white noise')
plt.show()


# STATIONARITY 

t_statistic, p_value, lags_number, n_observation, critical_values, max_info_criteria =sts.adfuller(df.DKe_wind)
# H0                Non stationarity
# t_statistic       If greater than critcal_values, not evidence of stationarity
# p_value           % of not rejecting H0
# lags_number       numebr of lags used in regression when performinf t-statistics
# n_observation     
# critical_values   
# max_info_crit     

# SEASONALITY 

df_decomp_additive = seasonal_decompose(df.DKe_wind, model='additive')
df_decomp_additive.plot()
plt.show()

# df_decomp_multiplicative = seasonal_decompose(df.DKe_wind, model='multiplicative')
# df_decomp_multiplicative.plot()
# plt.show()

# AUTOCORRELATION

sgt.plot_acf(df.SE4_wind, lags = 40, zero = False)
plt.title('ACF')
plt.show()

# PARTIAL AUTO CORRELATION FUNCTION

sgt.plot_pacf(df.SE4_wind, lags = 40, zero = False, method = ('ols'))
plt.title('PACF')
plt.show()