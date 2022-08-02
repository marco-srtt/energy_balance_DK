import numpy as np
import pandas as pd

import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.seasonal import seasonal_decompose
import calplot

import matplotlib.pyplot as plt


#%%
# IMPORT DATA
filename        = "market_data/fcrreservesdk2_09_12_2021.xlsx"
raw_csv_data    = pd.read_excel(filename)
df_FCR         = raw_csv_data.copy()

filename        = "market_data/balancing_2021.xlsx"
raw_bal_data    = pd.read_excel(filename)
df_bal          = raw_bal_data.copy()


#%%
# DATA PREPARATION

# FCR
AttributeNamesFCR  = list(df_FCR.columns.values)
df_FCR = df_FCR[list( AttributeNamesFCR[i] for i in [1, 3, 5] )]
# INDEX
df_FCR['HourDK'] = pd.to_datetime(df_FCR['HourDK'])
df_FCR.set_index('HourDK', inplace = True)   # inplace create an integer indexes
df_FCR.sort_index(inplace = True)
# BALANCING PRICES 
AttributeNamesBal  = list(df_bal.columns.values)
df_bal = df_bal[list( AttributeNamesBal[i] for i in [1, 2, 8, 10] )]

# INDEX
df_bal['HourDK'] = pd.to_datetime(df_bal['HourDK'])
df_bal.set_index('HourDK', inplace = True)   # inplace create an integer indexes
df_bal.sort_index(inplace = True)
df_bal_DK2 = df_bal[df_bal.PriceArea == 'DK2']
df_bal_DK2 = df_bal_DK2.drop('PriceArea', 1)
df_bal_DK1 = df_bal[df_bal.PriceArea == 'DK1']

#%%
# COMPLETE DATASET

merged         = df_FCR.merge(df_bal_DK2, how="left", left_index=True, right_index=True) 
df_comp_winter = merged['2021-01': '2021-02'].copy()
df_comp_summer = merged['2021-07': '2021-08'].copy()

df_comp_winter.to_excel('Dataframes_test/Dataframe_Jan_Feb.xlsx')
df_comp_summer.to_excel('Dataframes_test/Dataframe_Jul_Aug.xlsx')
#%%
# TIME SERIES ANALYSIS

# SEASONALITY
#results = seasonal_decompose(df_FCR.FCR_N_PriceEUR, model='additive', period=6000)
#results.plot()


# AUTOCORRELATION
#days = 3        # Numer of days for autocorrelation evaluation
#sgt.plot_acf(df_FCR.FCR_N_PriceEUR, lags = days*24, zero = False)
#plt.title('ACF')
#plt.show()

# PARTIAL AUTO CORRELATION FUNCTION
#sgt.plot_pacf(df_FCR.FCR_N_PriceEUR, lags = 1*24, zero = False, method = ('ols'))
#plt.title('PACF')
#plt.show()

#%%
# DATA VISUALIZATION FCR

# FREQUENCY
#df_comp = df_comp.asfreq('h')               # set frequency of data
df_FCR.FCR_N_PriceEUR.plot(figsize = (25,5))
df_FCR.FCR_N_PriceEUR.rolling(window =3*24).mean().plot()
plt.ylabel('FCR-N Prices')
plt.savefig('images/FCR_N_DK2.png', dpi=1000)
plt.legend()
plt.show()

df_FCR.plot(subplots=True)
plt.show()



z = 0
year = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021])
year_price = np.zeros((24, len(year)))

# PLOTTING HOURLY AVERAGE PER YEAR
for y in year:
    for i in range(24):
        value = df_FCR.FCR_N_PriceEUR[(df_FCR.index.hour==i)&(df_FCR.index.year==y)].mean()
        year_price[i, np.argwhere(year==y)] = value
    plt.step(np.arange(24), year_price[:, z], label = 'Year ' + str(y), where='post')
    z = z +1
plt.legend(loc = 'best', fontsize=7)
plt.xticks(ticks=np.arange(24))
plt.margins(0,0.1)
plt.xlabel('Hours')
plt.ylabel('FCR-N payment [€/MW]')
plt.tight_layout()
plt.savefig('images/average_price_y.png', dpi = 1000)
plt.show()

# PLOTTING HOURLY  AVERAGE PER MONTH
month = np.arange(12)
month_price = np.zeros((24, len(month)))
month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
z = 0
for m in month:
    for i in range(24):
        value = df_FCR.FCR_N_PriceEUR[(df_FCR.index.hour==i)&(df_FCR.index.month==(m+1))].mean()
        month_price[i, np.argwhere(month==m)] = value
    plt.step(np.arange(24), month_price[:, z], label = month_label[m], where='post')
    z = z +1
plt.legend(loc = 'best', fontsize=7, ncol=4)
plt.xticks(ticks=np.arange(24))
plt.margins(0,0.1)
plt.xlabel('Hours')
plt.ylabel('FCR-N payment [€/MW]')
plt.tight_layout()
plt.savefig('images/average_price_m.png', dpi = 1000)
plt.show()

# PLOTTING MONTHLY AVERAGE 

month_price_yearly = np.zeros((len(month), len(year)))
z = 0
for y in year:
    for m in range(12):
        value = df_FCR.FCR_N_PriceEUR[(df_FCR.index.month==(m+1))&(df_FCR.index.year==y)].mean()
        month_price_yearly[m, z] = value
    plt.step(np.arange(12), month_price_yearly[:, z], label = 'Year ' + str(y), where='post')
    z = z +1
plt.legend(loc = 'best', fontsize=7, ncol=4)
plt.xticks(np.arange(12), month_label, rotation = 30)
plt.margins(0,0.1)
plt.xlabel('Hours')
plt.ylabel('FCR-N payment [€/MW]')
plt.tight_layout()
plt.savefig('images/average_price_monthly.png', dpi = 1000)
plt.show()

month_price_all = np.zeros([12])
for i in range(12):
    month_price_all[i] = np.mean(month_price_yearly[i,:])
plt.step(np.arange(12), month_price_all, where='post', label='Average monthly price')
plt.xticks(np.arange(12), month_label, rotation = 30)
#plt.margins(0,0)
plt.xlabel('Months')
plt.ylabel('FCR-N payment [€/MW]')
plt.legend(loc = 'best' )
plt.tight_layout()
plt.savefig('images/average_price_monthly_all.png', dpi = 1000)
plt.show()


#%%

# DATA VISUALIZATION BALANCING PRICE

df_bal_DK2.BalancingPowerPriceUpEUR.plot(figsize = (25,5))
df_bal_DK2.BalancingPowerPriceUpEUR.rolling(window =3*24).mean().plot()
plt.ylabel('Up regulation Prices 2021 [EUR/MWh]')
plt.savefig('images/Bal_UP_DK2.png', dpi=1000)
plt.legend()
plt.show()

df_bal_DK2.BalancingPowerPriceDownEUR.plot(figsize = (25,5))
df_bal_DK2.BalancingPowerPriceDownEUR.rolling(window =3*24).mean().plot()
plt.ylabel('Down regulation Prices 2021 [EUR/MWh]')
plt.savefig('images/Bal_DOWN_DK2.png', dpi=1000)
plt.legend()
plt.show()
 
#%%

# PLOTTING YEARLY AVERAGE 

price_yearly = np.zeros([len(year), 2])
z = 0
for y, years in enumerate(year):
    value_FCRN = df_FCR.FCR_N_PriceEUR[(df_FCR.index.year==years)].mean()
    value_FCRD = df_FCR.FCR_D_UpPriceEUR[(df_FCR.index.year==years)].mean()
    price_yearly[y, 0] = value_FCRN
    price_yearly[y, 1] = value_FCRD
    
coord = 0.20
w     = 1.5*coord
edge  = 'black'
plt.bar(np.arange(7)-coord, price_yearly[:,0], width=w, edgecolor = edge, label='Average FCR-N yearly price')
plt.bar(np.arange(7)+coord, price_yearly[:,1], width=w, edgecolor = edge, label='Average FCR-D Up yearly price')
plt.xticks(np.arange(7), year)
#plt.margins(0,0)
plt.xlabel('Year')
plt.ylabel('Average capacity payment [€/MW]')
plt.legend(loc = 'best', fontsize = 8 )
plt.tight_layout()
plt.savefig('images/average_price_yearly_all.png', dpi = 1000)
plt.show()

#%%
# TEST
test  = df_FCR.resample('D').mean()
calplot.calplot(test.FCR_D_UpPriceEUR)
#plt.tight_layout()
plt.savefig('images/calendar.png', dpi = 1000)
