"""
Created on Thu Nov  4 11:40 2021

@author: Xenia Hassing-Hansen

Script for calculating how much 1 kg of H2 costs to produce at a given 
electrolysis plant size.
"""

from numpy import arange
import pandas as pd
import numpy_financial as npf
import matplotlib.pyplot as plt

'''
Parameters (for now, all the values are educated guesses (by Roar))
'''
#sz_elec = 4.8 * 1000           # kW        # Size of electrolysis plant
sz_elec_start   = 4200          # kW
sz_elec_end     = 5000          # kW
sz_elec_step    = 50            # kW
sz_wind         = 5.0 * 1000    # kW        # Size of wind farm
HHV             = 40            # kWh/kg    # Higher heating value
eff             = 0.90          # Efficiency in decimals
pow_price       = 0.04          # $/kWh     # Price of the wind power
H2_price        = 6             # $/kg      # Price of 1 kg green H2

# The values below are from Marco's "Heatpump" Excel document.
cost_stack      = 200   # $/kW      # Cost of electrolyzer stack
stack_lifetime  = 5     # years
cost_BoP        = 350   # $/kW      # Cost of BoP (balance of plant)
BoP_lifetime    = 20    # years     
lifespan        = 20    # years


# The values below descibe parmeters for wind farm CAPEX and OPEX and come from https://ens.dk/sites/ens.dk/files/Statistik/technology_data_catalogue_for_el_and_dh_-_0009.pdf p. 246
nom_investment  = 2012.5    # $/kW      # Nominal investment
OnM_fixed       = 41.461    # $/kW/year # Fixed O&M
OnM_var         = 0.0030705 # $/kWh     # Variable O&M
lifetime_wind   = 27        # years     # Technical lifetime

#input_list = [sz_elec_start, sz_wind, ] # Not finished - is it needed?

# Statement1: "When including the wind CAPEX and OPEX, we do not pay for the power"
# Statement2: "The excess wind power can be sold and the value subtracted to reduce the LCoH"
Statement1 = True
Statement2 = False

'''
Creating a for-loop to test varying sizes of electrolyzer plant and creating the lists SOEC_sz and LCoH_list
'''
SOEC_sz     = []
LCoH_list   = []
IRR_list    = []
NPV_list    = []
for sz_elec in list(range(sz_elec_start, (sz_elec_end + sz_elec_step), sz_elec_step)):
    
    SOEC_sz = SOEC_sz + [sz_elec]       # kW

    '''
    Importing wind data

    Note: The values in the columns DKe_wind, SE2_wind, and SE4_wind of 
    'wind_pow.csv' correspond to percentagewise amounts of peak wind at 
    three different locations, the first being in Denmark.
    I.e. they should be multiplied by the size of the wind farm, e.g. 1 MW.
    '''
    df = pd.read_csv('wind_pow_1month.csv')          # Two years
    #df = pd.read_csv(r'T:\01 - Hybrid Greentech ApS\07 - Research projects\06PF - NEXP2X\04_Assignments\Code\hydrogen-cost\wind_pow_1month.csv')    # One month

    '''
    Converting DD/MM/YYYY HH:MM to accumulated hours
    '''
    df['Time'] = pd.to_datetime(df['Time'], format="%d/%m/%Y %H:%M") #Converted to datetime format

    acc_time = []               # Accumulated time in hours
    for idx, row in df.iterrows():
        init_time = df.at[0, 'Time']
        curr_time = df.at[idx, 'Time']
        diff = (curr_time - init_time).total_seconds() / 3600
        acc_time = acc_time + [diff]
        #print(' Initial time:', init_time, '\nCurrent time:', curr_time, '\nDifference:', diff)
    df['Acc_time'] = acc_time

    '''
    Mass of produced H2 gas

    In the for-loop, the power produced by the wind farm, is initially 
    determined, and it is found whether this exceeds the capacity of the 
    electrolysis plant.
    '''
    mass_H2     = []
    input_pow   = []      # The amount of power the wind park "sends" to the electrolyzer plant
    used_pow    = []      # The amount of power the electrolyzer used of the input power
    for idx, row in df.iterrows():
        wind = sz_wind * (df.at[idx, 'DKe_wind'])
        elec = sz_elec
        #print(idx, 'wind=', wind, 'elec=', elec)
        if (wind - elec) >= 0:
            pow_used = elec
        if (wind - elec) < 0:
            pow_used = wind
        mass_H2 = mass_H2 + [(pow_used * 0.0233)]         # m = pow_used / HHV * efficiency
        # mass_H2 = mass_H2 + [(pow_used / HHV) * eff]  # m = pow_used / HHV * efficiency
        input_pow = input_pow + [wind]                  # Corresponds to wind profile
        used_pow = used_pow + [pow_used]                # The power that was actually used by the electrolyzer
    df['Mass_H2'] = mass_H2
    df['Input_pow'] = input_pow
    #print(df)

    unused_pow = [x1 - x2 for (x1, x2) in zip(input_pow, used_pow)]     # kWh       # Subtracting the values in used_pow from input_pow

    '''
    Printing the obtained values for the given system for inspection
    '''
    dataset_time_passed = df['Acc_time'].iat[-1] / (24 * 365.2425)      # years     # Choosing the last value from 'Acc_Time', i.e. the total amount of accumulated hours, and then converting it to years (taking leap years and "400 year exception" into account).
    multiplier = lifespan / dataset_time_passed                         # Functioning as a "time extender" extending the dataset from 1 month to 20 years
    time_passed = dataset_time_passed * multiplier                      # years     # Equals lifespan
    total_mass_H2 = df['Mass_H2'].sum() * multiplier                    # kg
    total_wind_pow = df['Input_pow'].sum() * multiplier                 # kWh

    # Paying for power or not
    if Statement1 == True:
        cost_electricity = 0
    else:
        cost_electricity = total_wind_pow * pow_price * multiplier     # $

    # CAPEX
    SOEC_investment = (cost_stack * sz_elec) * (lifespan / stack_lifetime) + (cost_BoP * sz_elec) * (lifespan / BoP_lifetime)       # Corresponding to C_investment = C_stack * t_sys_life / t_stack_life + C_BoP * t_sys_life / t_BoP_life from 'Heatpump.xlsx'
    remain_SOEC_value = (cost_stack * sz_elec) * ((lifespan - time_passed) / stack_lifetime) + (cost_BoP * sz_elec) * ((lifespan - time_passed) / BoP_lifetime)  # Remaining value of the electrolyzer plant at the time of the last wind power data point.
    wind_investment = nom_investment * sz_wind                                   # At some point this should maybe be multiplied by a factor lifespan_wind/turbine_lifetime, but I would need to discuss this with someone. Then also change remain_wind_value parameter.
    remain_wind_value = nom_investment * sz_wind * ((lifetime_wind - time_passed) / lifetime_wind)
    cost_investment = SOEC_investment + wind_investment
    remain_value = remain_SOEC_value + remain_wind_value
    CAPEX = cost_investment - remain_value                                       # $      # Based on CAPEX = C_investment / Q_H2_produced from 'Heatpump.xlsx', but subtracting the remaining value of the plant at the time of the last wind power data point.

    # OPEX
    OnM_SOEC = 0.02 * (cost_stack + cost_BoP) * sz_elec * time_passed            # SOEC operations and maintenance. The 0.02 is actually 2 %/year
    OnM_wind = (OnM_fixed * time_passed * sz_wind) + (OnM_var * total_wind_pow)  # Wind farm operations and maintenance 
    OnM = OnM_SOEC + OnM_wind                                                    # Total operations and maintenance value
    OPEX = cost_electricity + OnM                                                # $      # In 'Heatpump.xlsx' it is written as OPEX = C_electricity / Efficiency + O&M -- Do I already take Eff into account at an earlier stage?

    # Income from unused power or not
    if Statement2 == True:
        income_unused_pow = sum(unused_pow) * multiplier * pow_price
    else:
        income_unused_pow = 0

    #print(income_unused_pow)

    # Levelized cost of H2 (LCoH)
    LCoH = (CAPEX + OPEX - income_unused_pow) / total_mass_H2                    # $/kg H2
    LCoH_list = LCoH_list + [LCoH]
    print('\nSOEC:', sz_elec / 1000, 'kW, Wind farm:', sz_wind / 1000, 'kW \n', 'Total mass of H2:\t', '%.1f' % total_mass_H2, 'kg \n', 'Total power used:\t', '%.1f' % total_wind_pow, 'kWh \n', 'Total power cost: \t', '%.1f' % cost_electricity, '$ \n', 'Total investment cost:\t', '%.1f' % CAPEX, '$ \n', 'Cost of H2 production:\t', '%.2f' % LCoH, '$/kg H2')

    ''''
    Calculating the Internal Rate of Return (IRR) and Net Present Value (NPV)
    
    This includes a repetition of code from above -> perhaps make it into functions at some point instead.
    '''
    inflation = 0.02
    discount_rate = 0.04

    period_sz = 1                                                       # year(s)
    period_number = arange(0, lifespan, period_sz)
    yrly_multiplier = period_sz / dataset_time_passed                   # Functioning as a "time extender" extending the dataset from 1 month to 1 year
    yrly_time_passed = dataset_time_passed * yrly_multiplier            # years     # Equals period_sz
    yrly_mass_H2 = df['Mass_H2'].sum() * yrly_multiplier                # kg
    yrly_wind_pow = df['Input_pow'].sum() * yrly_multiplier             # kWh

    # Paying for power or not
    if Statement1 == True:
        yrly_cost_electricity = 0
    else:
        yrly_cost_electricity = yrly_wind_pow * pow_price * yrly_multiplier  # $

    # Income from unused power or not
    if Statement2 == True:
        income_unused_pow = sum(unused_pow) * yrly_multiplier * pow_price
    else:
        income_unused_pow = 0

    # Yearly OPEX
    OnM_SOEC = 0.02 * (cost_stack + cost_BoP) * sz_elec * yrly_time_passed              # SOEC operations and maintenance. The 0.02 is actually 2 %/year
    OnM_wind = (OnM_fixed * yrly_time_passed * sz_wind) + (OnM_var * yrly_wind_pow)     # Wind farm operations and maintenance 
    OnM = OnM_SOEC + OnM_wind                                                           # Total operations and maintenance value
    yrly_OPEX = yrly_cost_electricity + OnM                                             # $      # In 'Heatpump.xlsx' it is written as OPEX = C_electricity / Efficiency + O&M -- Do I already take Eff into account at an earlier stage?

    # Creating a list of the net cash flow for the years from t=0 to T=20.
    for t in period_number:
        if t == 0:
            net_cf = [-CAPEX]
        else:
            net_cf = net_cf + [(yrly_mass_H2 * H2_price + income_unused_pow - yrly_OPEX) * (1 + inflation)**t]

    IRR_list = IRR_list + [npf.irr(net_cf)]

    # Calculating the NPV:
    discounted_cf = []              # List of the discounted cash flows
    for t in period_number:
        discounted_cf = discounted_cf + [net_cf[t] / (1 + discount_rate)**t]
    NPV = sum(discounted_cf)
    #print(discounted_cf)
    NPV_list = NPV_list + [NPV]


eval = pd.DataFrame({'SOEC size': SOEC_sz, 'LCoH': LCoH_list, 'IRR': IRR_list, 'NPV': NPV_list})

'''
Plotting the development in different parameters
'''
#print('LCoH:', LCoH_list, '\nfor SOECs of sizes:', SOEC_sz)
print(eval)
    
fig, (ax1, ax2) = plt.subplots(2, sharex=True)    # Top subplot
color = 'tab:green'
#ax1.set_xlabel("Size of SOEC plant [MW]")
ax1.set_ylabel("LCoH [$/kg H2]", color=color)
ax1.set_title("Evaluation parameters for different SOEC sizes")
ax1.plot([i / 1000 for i in SOEC_sz], LCoH_list, '.-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'       # Bottom subplot
ax2.set_xlabel("Size of SOEC plant [MW]")
ax2.set_ylabel("NPV [M$/kg H2]", color=color)
ax2.plot([i / 10**3 for i in SOEC_sz], [i / 10**6 for i in NPV_list], '.-', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax2.twinx()       # Second axis that shares x-axis with bottom plot

color = 'tab:blue'
ax3.set_ylabel("IRR [%]", color=color)
ax3.plot([i / 1000 for i in SOEC_sz], [i * 100 for i in IRR_list], '.-', color=color)
ax3.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # Otherwise the right y-label is slightly clipped
plt.savefig('results.png', dpi=1500)
plt.show()
#plt.plot(acc_time, input_pow, label='Input power')
#plt.plot(acc_time, used_pow, "--", label='Power used')
#plt.legend()
#plt.show()


'''
--------------------------------------------------------------------------------------------------
-------------------------------------------Functions----------------------------------------------
--------------------------------------------------------------------------------------------------
'''

def costelectricity(timeperiod):
    ''''''
    # Give the function a different name or remove the variables with the same name.
    # Just copypaste the if-statements from above and return cost_electricity.

def incomeunusedpower(timeperiod):
    # The same as costelectricity (see above)
    return

def CAPEX_f(lifespan):
    ''' 
    The variable 'input_list' must include wind and electrolyzer size, parameters for investment cost (stack cost, BoP cost, wind costs),
    lifetime of the components etc.         # Is this in fact needed if only the parameters are defined outside the function?
    '''
    # CAPEX
    SOEC_investment = (cost_stack * sz_elec) * (lifespan / stack_lifetime) + (cost_BoP * sz_elec) * (lifespan / BoP_lifetime)       # Corresponding to C_investment = C_stack * t_sys_life / t_stack_life + C_BoP * t_sys_life / t_BoP_life from 'Heatpump.xlsx'
    remain_SOEC_value = (cost_stack * sz_elec) * ((lifespan - time_passed) / stack_lifetime) + (cost_BoP * sz_elec) * ((lifespan - time_passed) / BoP_lifetime)  # Remaining value of the electrolyzer plant at the time of the last wind power data point.
    wind_investment = nom_investment * sz_wind                                   # At some point this should maybe be multiplied by a factor lifespan_wind/turbine_lifetime, but I would need to discuss this with someone. Then also change remain_wind_value parameter.
    remain_wind_value = nom_investment * sz_wind * ((lifetime_wind - time_passed) / lifetime_wind)
    cost_investment = SOEC_investment + wind_investment
    remain_value = remain_SOEC_value + remain_wind_value
    CAPEX = cost_investment - remain_value                                       # $      # Based on CAPEX = C_investment / Q_H2_produced from 'Heatpump.xlsx', but subtracting the remaining value of the plant at the time of the last wind power data point.
    return CAPEX

def OPEX_f(timeperiod):
    '''
    To calculate the OPEX for a given timeperiod. 
    'timeperiod' could be equal to, e.g., 'lifespan' or 'period_sz'.
    '''
    # Not defined yet!
    return

def multiplier(timeperiod):
    '''
    To calculate the value that must be multiplied to different parameters in order for them to describe the entire time period and not just
    the length of the RE dataset.
    'timeperiod' could be equal to, e.g., 'lifespan' or 'period_sz'.
    '''
    # Not defined yet!
    return