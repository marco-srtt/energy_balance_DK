import requests
import pandas as pd

headers = {'content-type': 'application/json'}

query = """
{
  elspotprices(limit: 50000, order_by : {desc : HourDK}) {
    
    HourUTC
    
    HourDK
    
    PriceArea
    
    SpotPriceDKK
    
    SpotPriceEUR
    
  }
}
"""

request     = requests.post('https://data-api.energidataservice.dk/v1/graphql', json={'query': query}, headers=headers)
as_json     = request.json()
elspot_api  = as_json.get('data').get('elspotprices')
#%% DATAFRAME CREATION

# SELECTING COLUMNS
raw_data    = pd.DataFrame(elspot_api)
elspot      = raw_data[['HourDK', 'PriceArea', 'SpotPriceEUR']].copy()


# INDEX SORTING TO DATETIME
elspot.HourDK   = pd.to_datetime(elspot.HourDK)
elspot.set_index('HourDK', inplace = True)
elspot  = elspot.sort_index(axis=0, ascending=True)


# PRINT DK2 PRICES FOR YEAR 2019 
DK2_2019 = elspot[(elspot.PriceArea == 'DK2') & (elspot.index.year == 2019) ]
print(DK2_2019)