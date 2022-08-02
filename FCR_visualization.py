import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
#%% IMPORTING RAW DATA

# Dataset for historical price of FCR privision per MW
raw_data_DK1 = pd.read_excel('Data/FCR/FcrReservesDK1.xlsx')
raw_data_DK2 = pd.read_excel('Data/FCR/FcrReservesDK2.xlsx')

# FCR
AttributeNamesFCR  = list(raw_data_DK2.columns.values)
df_FCR = raw_data_DK2[list( AttributeNamesFCR[i] for i in [1, 3, 5] )]
#%% DATA PROCESSING

# INDEX
# Parsing dates in the correct format
df_FCR['HourDK'] = pd.to_datetime(df_FCR['HourDK'])
df_FCR.set_index('HourDK', inplace = True)   # inplace create an integer indexes
df_FCR.sort_index(ascending = False, inplace = True)

#%% CUMULATIVE PAYMENT
df_FCR['year'] = df_FCR.index.year
df_FCR['month'] = df_FCR.index.month
df_FCR['week'] = df_FCR.index.week
df_FCR['weekday'] = df_FCR.index.weekday
df_FCR['day'] = df_FCR.index.day
df_FCR['hour'] = df_FCR.index.hour
year_FCR = df_FCR.groupby(by = 'year').sum()
N = len(year_FCR.columns)
ind = np.arange(N) 
width = 0.35
space = width * 0.55


fig, ax = plt.subplots(figsize=(10,5))


bar_FCRN = ax.bar(year_FCR.index -space, year_FCR.FCR_N_PriceEUR, width, color='#088da5')

bar_FCRD = ax.bar(year_FCR.index +space, year_FCR.FCR_D_UpPriceEUR, width, color='#194366')
  
plt.xlabel("Year")
plt.ylabel('FCR payment [€/MW]')
plt.title("Cumulative payment")
  

plt.legend( (bar_FCRN, bar_FCRD), ('FCR-N', 'FCR-D Up') )
plt.tight_layout()
plt.savefig('Figures/barchart.png', dpi = 600)
plt.show()

#%% LINEPLOT
linewidth = 0.5
fig, ax = plt.subplots(figsize=(10 ,5))

fcrn_line = ax.plot(df_FCR.index , df_FCR.FCR_N_PriceEUR, linewidth = linewidth, color='#088da5', label = 'FCR-N regulation price [€/MW]')
fcrd_line = ax.plot(df_FCR.index , df_FCR.FCR_D_UpPriceEUR, linewidth = linewidth, color='#194366', label = 'FCR-D Up regulation price [€/MW]')

plt.xlabel("Year")
plt.ylabel('FCR payment [€/MW]')
plt.title("Frequency Containment Reserves prices DK2")
  

plt.legend( )
plt.tight_layout()
plt.savefig('Figures/lineplot.png', dpi = 600)
plt.show()

#%% PIVOT MONTH
pivot_month_FRCN = df_FCR.pivot_table(index='year', columns='month', values='FCR_N_PriceEUR')
pivot_month_FRCD = df_FCR.pivot_table(index='year', columns='month', values='FCR_D_UpPriceEUR')

#import matplotlib.colors
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [ '#194366',"#088da5", '#ffe300'])
#sns.heatmap(pivot_month, cmap='viridis')
months = ['Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, (axN, axD) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize = (15, 5))
for i in range(len(pivot_month_FRCN)):
    axN.step(months, pivot_month_FRCN.iloc[i,:], label = 'Year ' + str(pivot_month_FRCN.index[i]), where ='post')
    axD.step(months, pivot_month_FRCD.iloc[i,:], label = 'Year ' + str(pivot_month_FRCD.index[i]), where ='post')
axN.legend(ncol = 2, fontsize = 8)
axN.set_xlabel('Months')
axN.set_ylabel('Average monthly FCR-N price [€/MW]')
axD.legend(ncol = 2, fontsize = 8)
axD.set_xlabel('Months')
axD.set_ylabel('Average monthly FCR-D Up price [€/MW]')
axD.yaxis.set_tick_params(labelbottom=True)
plt.tight_layout()
plt.savefig('Figures/pivot_month.png', dpi = 600)
plt.show()
#%% PIVOT DAY
pivot_day_FRCN = df_FCR.pivot_table(index='year', columns='day', values='FCR_N_PriceEUR')
pivot_day_FRCD = df_FCR.pivot_table(index='year', columns='day', values='FCR_D_UpPriceEUR')

fig, (axN, axD) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize = (15, 5))
for i in range(len(pivot_day_FRCN)):
    axN.step(range(31), pivot_day_FRCN.iloc[i,:], label = 'Year ' + str(pivot_day_FRCN.index[i]), where ='post')
    axD.step(range(31), pivot_day_FRCD.iloc[i,:], label = 'Year ' + str(pivot_day_FRCD.index[i]), where ='post')
axN.legend(ncol = 2, fontsize = 8)
axN.set_xlabel('days')
axN.set_ylabel('Average dayly FCR-N price [€/MW]')
axD.legend(ncol = 2, fontsize = 8)
axD.set_xlabel('Day')
axD.set_ylabel('Average dayly FCR-D Up price [€/MW]')
axD.yaxis.set_tick_params(labelbottom=True)

# Setting the xlabel ticks for the FCRD plot
axD.set_xticks(range(31))
labels = np.arange(1, 32)
axD.set_xticklabels(labels)

# Setting the xlabel ticks for the FCRN plot
axN.set_xticks(range(31))
labels = np.arange(1, 32)
axN.set_xticklabels(labels)
plt.savefig('Figures/pivot_hour.png', dpi = 600)
plt.tight_layout()
plt.show()
#%% PIVOT WEEK
pivot_week_FRCN = df_FCR.pivot_table(index='year', columns='week', values='FCR_N_PriceEUR')
pivot_week_FRCD = df_FCR.pivot_table(index='year', columns='week', values='FCR_D_UpPriceEUR')

fig, (axN, axD) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize = (15, 5))
for i in range(len(pivot_week_FRCN)):
    axN.step(range(53), pivot_week_FRCN.iloc[i,:], label = 'Year ' + str(pivot_week_FRCN.index[i]), where ='post')
    axD.step(range(53), pivot_week_FRCD.iloc[i,:], label = 'Year ' + str(pivot_week_FRCD.index[i]), where ='post')
axN.legend(ncol = 2, fontsize = 8)
axN.set_xlabel('Week')
axN.set_ylabel('Average weekly FCR-N price [€/MW]')
axD.legend(ncol = 2, fontsize = 8)
axD.set_xlabel('Week')
axD.set_ylabel('Average weekly FCR-D Up price [€/MW]')
axD.yaxis.set_tick_params(labelbottom=True)

# Setting the xlabel ticks for the FCRD plot
axD.set_xticks(range(0, 53))
labels = np.arange(1, 54)
axD.set_xticklabels(labels, fontsize = 8)

# # Setting the xlabel ticks for the FCRN plot
axN.set_xticks(range(0, 53))
labels = np.arange(1, 54)
axN.set_xticklabels(labels, fontsize = 8)

plt.savefig('Figures/pivot_hour.png', dpi = 600)
plt.tight_layout()
plt.show()
#%% PIVOT weekdayDAY
pivot_weekday_FRCN = df_FCR.pivot_table(index='year', columns='weekday', values='FCR_N_PriceEUR')
pivot_weekday_FRCD = df_FCR.pivot_table(index='year', columns='weekday', values='FCR_D_UpPriceEUR')

fig, (axN, axD) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize = (15, 5))

weedays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(len(pivot_weekday_FRCN)):
    axN.step(weedays, pivot_weekday_FRCN.iloc[i,:], label = 'Year ' + str(pivot_weekday_FRCN.index[i]), where ='post')
    axD.step(weedays, pivot_weekday_FRCD.iloc[i,:], label = 'Year ' + str(pivot_weekday_FRCD.index[i]), where ='post')
axN.legend(ncol = 2, fontsize = 8)
axN.set_xlabel('Weekday')
axN.set_ylabel('Average weekdayly FCR-N price [€/MW]')
axD.legend(ncol = 2, fontsize = 8)
axD.set_xlabel('Weekday')
axD.set_ylabel('Average weekdayly FCR-D Up price [€/MW]')
axD.yaxis.set_tick_params(labelbottom=True)


plt.savefig('Figures/pivot_weekday.png', dpi = 600)
plt.tight_layout()
plt.show()
#%% PIVOT HOUR

# Defining pivot values
pivot_hour_FRCN = df_FCR.pivot_table(index='year', columns='hour', values='FCR_N_PriceEUR')
pivot_hour_FRCD = df_FCR.pivot_table(index='year', columns='hour', values='FCR_D_UpPriceEUR')

# Creating subplots
fig, (axN, axD) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize = (15, 5))
for i in range(len(pivot_hour_FRCN)):
    axN.step(range(24), pivot_hour_FRCN.iloc[i,:], label = 'Year ' + str(pivot_hour_FRCN.index[i]), where ='post')
    axD.step(range(24), pivot_hour_FRCD.iloc[i,:], label = 'Year ' + str(pivot_hour_FRCD.index[i]), where ='post')

# Legend settings
axN.legend(ncol = 2, fontsize = 8)
axN.set_xlabel('Hour')
axN.set_ylabel('Average hourly FCR-N price [€/MW]')
axD.legend(ncol = 2, fontsize = 8)
axD.set_xlabel('Hour')
axD.set_ylabel('Average hourly FCR-D Up price [€/MW]')
axD.yaxis.set_tick_params(labelbottom=True)

# Setting the xlabel ticks for the FCRD plot
axD.set_xticks(range(24))
labels = np.arange(1, 25)
axD.set_xticklabels(labels)

# Setting the xlabel ticks for the FCRN plot
axN.set_xticks(range(24))
labels = np.arange(1, 25)
axN.set_xticklabels(labels)
plt.savefig('Figures/pivot_hour.png', dpi = 600)
plt.tight_layout()
plt.show()

