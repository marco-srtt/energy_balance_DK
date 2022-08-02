import requests as req
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels as sm


# DATAFRAME SPOT MARKET
# Real time market contains information of spot price
# Info at https://www.energidataservice.dk/tso-electricity/elspotprices

url: str = "https://api.energidataservice.dk/datastore_search_sql?sql={sqlFormat}"
query = """SELECT * from "elspotprices" where "HourDK" >= '2019-01-01 00:00:00' and "HourDK" < '2022-01-21 00:00:00'"""
resp = req.get(url.format(sqlFormat=query))
print('loading done 1')

result      = resp.json().get("result")
raw_data    = pd.DataFrame(result.get("records"))


#%% DATA PREPARATION

# SELECTING COLUMNS
# ['HourUTC', 'HourDK', '_full_text', 'FCR_D_UpPriceEUR', 'FCR_D_UpPriceDKK', 'FCR_N_PriceEUR', '_id', 'FCR_N_PriceDKK']
col_imported = [raw_data.columns[i] for i in [0, 1, 5]]
elspot = raw_data[col_imported] 

elspot.HourUTC = pd.to_datetime(elspot.HourUTC)
elspot.sort_values('HourUTC', inplace=True)
# FORMATTING INDEX
elspot['HourUTC'] = pd.to_datetime(elspot['HourUTC'])
elspot.sort_values('HourUTC',inplace=True)
elspot.set_index('HourUTC' ,inplace=True)

# REFERENCE COLUMNS
elspot['dates'] = elspot.HourUTC.dt.date.astype('str')
elspot['weeks'] = elspot.HourUTC.dt.week
elspot['hours'] = elspot.HourUTC.dt.hour
elspot['years'] = elspot.HourUTC.dt.year

elspot_DK2 = elspot[elspot.PriceArea=='DK2']
elspot_DK1 = elspot[elspot.PriceArea=='DK1']
elspot_SE3 = elspot[elspot.PriceArea=='SE3']
elspot_SE4 = elspot[elspot.PriceArea=='SE4']
elspot_NO2 = elspot[elspot.PriceArea=='NO2']
elspot_DE  = elspot[elspot.PriceArea=='DE']
elspot_SYS = elspot[elspot.PriceArea=='SYSTEM']

#%%

# TASK 2

years = [2019, 2020, 2021]

for i in years:
    avg_price  = elspot_DK2[(elspot_DK2.index.year == i)].mean()
    print(avg_price)