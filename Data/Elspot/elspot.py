import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
raw_data_DK1 = pd.read_csv('data/elspotprices_DK1.csv')
raw_data_DK2 = pd.read_csv('data/elspotprices_DK2.csv')

elspot_DK1 = raw_data_DK1[['HourDK', 'PriceArea', 'SpotPriceEUR']].copy()
elspot_DK2 = raw_data_DK2[['HourDK', 'PriceArea', 'SpotPriceEUR']].copy()

elspot_DK1.HourDK   = pd.to_datetime(elspot_DK1.HourDK)
elspot_DK2.HourDK   = pd.to_datetime(elspot_DK2.HourDK)

elspot_DK1.set_index('HourDK', inplace = True)
elspot_DK2.set_index('HourDK', inplace = True)

elspot_DK1 = elspot_DK1.sort_index(axis=0, ascending=True)
elspot_DK2 = elspot_DK2.sort_index(axis=0, ascending=True)

year     = np.linspace(2015, 2022, 8)
mean_DK1 = np.zeros(8)
mean_DK2 = np.zeros(8)

for i, years in enumerate(year):
    mean_DK1[i] = elspot_DK1[ (elspot_DK1.index.year == years) ].SpotPriceEUR.mean()
    mean_DK2[i] = elspot_DK2[ (elspot_DK2.index.year == years) ].SpotPriceEUR.mean()
    
mean = pd.DataFrame()
mean['year'] = year
mean['mean_DK1'] = mean_DK1
mean['mean_DK2'] = mean_DK2

mean.year = pd.to_datetime(mean.year, format='%Y')
mean.set_index('year', inplace = True)


fig, ax = plt.subplots(figsize=(10,4))
ax.plot(mean.mean_DK1, marker = '.', label='DK1 average spot price')
#ax.plot(elspot_DK1[(elspot_DK1.index.day == 1].SpotPriceEUR)
ax.plot(mean.mean_DK2, marker = '.', label='DK2 average spot price')
ax.set_ylabel('Spot price [€/MWh]')
ax.set_xlabel('Year')
ax.legend(loc = 'best')
ax.grid()
plt.savefig('Spot price.png', dpi=600)
plt.show()
