import requests
import pandas as pd
import matplotlib.pyplot as plt
headers = {'content-type': 'application/json'}

query = """
{
  fcrreservesdk2(limit: 10000) {
    
    HourDK
    
    FCR_N_PriceDKK
    
    FCR_N_PriceEUR
    
    FCR_D_UpPriceDKK
    
    FCR_D_UpPriceEUR
    
  }
}
"""

request = requests.post('https://data-api.energidataservice.dk/v1/graphql', json={'query': query}, headers=headers)
as_json = request.json()

fcr_data = as_json.get('data').get('fcrreservesdk2')

dfcomp = pd.DataFrame(fcr_data)

dfcomp.HourDK   = pd.to_datetime(dfcomp.HourDK)
dfcomp.set_index('HourDK', inplace = True)

dfcomp = dfcomp.sort_index(axis=0, ascending=True)
#%%
# DATA VISUALIZATION

fig, ax = plt.subplots(2, 1, figsize=(10,8), sharey=True)

fcr_N = dfcomp.FCR_N_PriceEUR
moving = dfcomp.FCR_N_PriceEUR.rolling(window =3*24).mean()
ax[0].plot( fcr_N, label = 'FCR-N Capacity payments')
ax[0].plot( moving, label = '72 hours moving average')
ax[0].legend(loc = 'best')
ax[0].set_ylabel('FCR-N Capacity payment [€/MW]')

fcr_D = dfcomp.FCR_D_UpPriceEUR
moving_D = dfcomp.FCR_D_UpPriceEUR.rolling(window =3*24).mean()

ax[1].plot( fcr_D, label = 'FCR-D Capacity payments')
ax[1].plot( moving_D, label = '72 hours moving average')
ax[1].legend(loc = 'best')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('FCR-D Capacity payment [€/MW]')
plt.savefig('images/FCR_N_DK2.png', dpi=1000)
plt.show()














