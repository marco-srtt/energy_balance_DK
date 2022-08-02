"""
Created on Thu Nov  4 11:40 2021

@author: Xenia Hassing-Hansen

Script for determining which electrolysis plant size is the most favorable 
for a given wind farm size. The input can be manipulated, and results are 
visualized in a browser using Streamlit. 
"""

import streamlit as st
import time
import datetime as dt
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from functions.func_data_streamlit_numpy import *

st.title("NEXP2X - Power-to-X Software Platform")
st.header("For market and business case analysis on hydrogen production via SOECs")
st.write(
    "In order to have an influence on the calculation, open the sidebar to the left, and modify the parameters to match your case."
)

start1 = time.time()

# '''
# --------------------------- Progress bar --------------------------
# '''
st.subheader("Calculation progress")
progress_bar = st.progress(0)
progress_step = 1 / (4)
percent_complete = 0

# '''
# ----------------------- Importing wind data -----------------------
# '''

folderpath = (
    r"T:\01 - Hybrid Greentech ApS\07 - Research projects\06PF - NEXP2X\04_Assignments\Code\hydrogen-cost"
    )

choice_power_source = st.sidebar.radio(
    "Where will you get your electricity from?",
    ["The grid", "Building a wind farm"],
    index=1)

if choice_power_source == "Building a wind farm":
    # Loading the input data. In the app, the user can choose whether to perform 
    # the calculation using the small or the large dataset.
    if st.sidebar.checkbox("Use large dataset?"):
        filename = r"wind_pow.csv"
        dataset_size = "2 years"
    else:
        filename = r"wind_pow_1month.csv"  # One month
        dataset_size = "1 month"
    filepath = folderpath + '\\' + filename
    bidding_zone = np.nan
    data = load_data(filepath, bidding_zone)

if choice_power_source == "The grid":
    # Loading hourly Elspot prices from 2015.
    filename = r"elspot-prices_2015_hourly_DKK.xlsx"
    dataset_size = "1 year (2015) "
    filepath = folderpath + '\\' + filename

    bidding_zone = st.sidebar.radio(
                    "Which bidding zone are you in?",
                    ["DK1", "DK2"],
                    index=1)

    data = load_data(filepath, bidding_zone)  # Accumulated time in first column, elspot prices in second column.

# '''
# ----------------------- Input parameters -----------------------
# (See Assumptions-powerpoint for references of the values)
# '''
st.sidebar.header("Input parameters")

st.sidebar.subheader("Sizes")

if choice_power_source == "Building a wind farm":
    sizes = building_windfarm_input()
    sz_wind = sizes[3]
    
if choice_power_source == "The grid":
    sizes = grid_power_source_input()
    sz_wind = 0

sz_elec_start = sizes[0]
sz_elec_end = sizes[1]
sz_elec_step = sizes[2]

# HHV = 39.41                   # kWh/kg    # Higher heating value
# eff = 0.90                                # Efficiency in decimals
LHV_H2 = 33.33                  # kWh/kg
LHV_MeOH = 5.54                 # kWh/kg

H2_output_factor = 0.0233       # kg/kWh input
pow_price = 0.04                # $/kWh     # Price of the wind power
H2_price = 6                    # $/kg      # Price of 1 kg green H2
excess_heat_price = 0.010872    # $/kWh     # Income from selling excess heat to district heating
mean_FCR = 0.575                # $/kW/day  # Mean income from contributing to the FCR per kW per day

# The values below are from Marco's "Heatpump" Excel document.
cost_stack = 200                # $/kW      # Cost of electrolyzer stack
stack_lifetime = 5              # years
cost_BoP = 350                  # $/kW      # Cost of BoP (balance of plant)
BoP_lifetime = 20               # years
lifespan = 20                   # years

# The values below descibe parmeters for wind farm CAPEX and OPEX and come from https://ens.dk/sites/ens.dk/files/Statistik/technology_data_catalogue_for_el_and_dh_-_0009.pdf p. 246
cost_windfarm = 2012.5          # $/kW      # Nominal investment
OnM_fixed = 41.461              # $/kW/year # Fixed O&M
OnM_var = 0.0030705             # $/kWh     # Variable O&M
windfarm_lifetime = 27          # years     # Technical lifetime

# Parameters related to loans and for describing where the money for the investment comes from.
st.sidebar.subheader("Investment related")

share_mortgage = (
    st.sidebar.slider(
        "Share of the initial investment (CAPEX) coming from a mortgage credit loan (in %)",
        0,
        100,
        value=15)
    / 100)  # Share of the initial investment (CAPEX) paid via a mortgage credit loan.

default_share_investment = 10

if (100 - share_mortgage * 100) < 10:
    default_share_investment = 1

share_investment = (st.sidebar.slider(
        "Share of the initial investment (CAPEX) coming from an investor loan (in %)",
        0,
        int(100 - share_mortgage * 100),
        value=default_share_investment) 
    / 100)  # Share of the initial investment (CAPEX) paid by a loan from an investor.

share_selfpayment = 1 - share_mortgage - share_investment  # Share of the initial investment (CAPEX) paid through self-payments.

# Bar plot indicating shares
labels = ["Mortgage credit loan", "Investor loan", "Self-payment"]
sizes = [share_mortgage * 100, share_investment * 100, share_selfpayment * 100]
fig1, ax1 = plt.subplots()
ax1.set_title("Distribution of the shares [%]")
for item in [fig1, ax1]:
    item.patch.set_visible(False)       # Removes background and axes from plot.
ax1.axis("off")

p1 = ax1.bar(1, sizes[0], label=labels[0])
p2 = ax1.bar(1, sizes[1], bottom=sizes[0], label=labels[1])
p3 = ax1.bar(1, sizes[2], bottom=sizes[1] + sizes[0], label=labels[2])
p = p1, p2, p3
for idx in range(0, len(p)):
    if sizes[idx] > 0:
        ax1.bar_label(p[idx], label_type="center")
ax1.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))

st.sidebar.pyplot(fig1)

# Yearly interest of the mortgage credit loan.
interest_mortgage = (
    st.sidebar.number_input("Interest of mortgage credit loan (in %)", 0, 100, value=1)
    / 100)  

# Yearly interest of the investor loan.
interest_invest = (
    st.sidebar.number_input("Interest of investor loan (in %)", 0, 100, value=7) / 100)  

# Dictionary of the various input parameters relevant for calculating different 
# investment variables. Useful in functions.
invest_params = {
    "sz_wind": sz_wind,
    "pow_price": pow_price,
    "excess_heat_price": excess_heat_price,
    "cost_stack": cost_stack,
    "cost_BoP": cost_BoP,
    "cost_windfarm": cost_windfarm,
    "stack_lifetime": stack_lifetime,
    "BoP_lifetime": BoP_lifetime,
    "windfarm_lifetime": windfarm_lifetime,
    "OnM_fixed": OnM_fixed,
    "OnM_var": OnM_var,
    "lifespan": lifespan,
    "mean_FCR": mean_FCR,
    "H2_output_factor": H2_output_factor}  

st.sidebar.subheader("Additional possible sources of income")

# Statement1: "The excess heat can be sold and the value subtracted to reduce the LCoH"
choice1 = st.sidebar.radio(
    "Can the excess heat be sold to district heating and the value subtracted to reduce the LCoH?",
    ["Yes", "No"],
    index=1)

Statement1 = (
    choice1 == "Yes")

# Statement2: "The excess wind power can be sold and the value subtracted to reduce the LCoH"
choice2 = st.sidebar.radio(
    "Can the excess wind power be sold and the value subtracted to reduce the LCoH?",
    ["Yes", "No"],
    index=1)

Statement2 = (
    choice2 == "Yes")

# Statement3: "Income is generated from contributing to the FCR (Frequency Containment Reserve)"
# OBS! For now, the function belonging to this is flawed. A fix could be figuring 
# out how much power is on average contributed with for one day and withdraw that 
# power from either used_pow or unused_pow.
choice3 = st.sidebar.radio(
    "Is income generated from contributing to the FCR (Frequency Containment Reserve)?",
    ["Yes", "No"],
    index=1)

Statement3 = (
    choice3 == "Yes")  

# Progress bar step
percent_complete = percent_complete + progress_step
progress_bar.progress(round(percent_complete, 3))

# ---------------------------------- End of input ----------------------------------
# ----------------------------------------------------------------------------------
# ---------------------------------- Calculations ----------------------------------

# Setting up array with varying sizes of the electrolyzer plant.
sz_elec = np.arange(sz_elec_start, (sz_elec_end + sz_elec_step), sz_elec_step)
invest_params["sz_elec"] = sz_elec

# Mass of produced H2 gas, and power used and input
mass_and_pow_values = mass_and_pow(data, choice_power_source, **invest_params)
dataset_mass_H2 = mass_and_pow_values[0]
dataset_input_pow = mass_and_pow_values[1]
dataset_used_pow = mass_and_pow_values[2]

# Progress bar step
percent_complete = percent_complete + progress_step
progress_bar.progress(round(percent_complete, 3))

# Array with total mass of H2 for the different SOEC sizes
total_mass_H2 = np.array(
    [dataset_mass_H2[:, i].sum() * multiplier(data, lifespan)
     for i in range(len(dataset_mass_H2[0]))])  # kg       

# Yearly payments related to the investment
payment_mortgage = installment_pay(lifespan, interest_mortgage, share_mortgage, **invest_params)
payment_invest = installment_pay(lifespan, interest_invest, share_investment, **invest_params)

# '''
# --------- Calculating the CAPEX, OPEX, and levelized cost of H2 (LCoH) ---------
# '''
CAPEX = CAPEX_f(lifespan, **invest_params)

OPEX = (
    OPEX_f(data, choice_power_source, lifespan, dataset_input_pow, **invest_params)
    )

LCoH = (
    CAPEX
    + OPEX
    + payment_mortgage
    + payment_invest
    - income_unused_pow(
        data, choice_power_source, Statement2, lifespan, dataset_input_pow, dataset_used_pow, **invest_params
    )
    - income_FCR(Statement3, lifespan, **invest_params)
    - income_excess_heat(data, Statement1, lifespan, dataset_used_pow, **invest_params)
    ) / total_mass_H2  # $/kg H2

# Printing the obtained values (in the terminal) for the varying SOEC sizes.
for i in range(len(sz_elec)):
    print("\nSOEC:",
        sz_elec[i] / 1000,
        "kW, Wind farm:",
        sz_wind / 1000,
        "kW \n",
        "Total mass of H2:\t",
        "%.1f" % total_mass_H2[i],
        "kg \n",
        "Total power used:\t",
        "%.1f" % (sum(dataset_used_pow[:, i]) * multiplier(data, lifespan)),
        "kWh \n",
        "Total power cost: \t",
        "%.1f"
        % cost_electricity(
            data, choice_power_source, lifespan, dataset_input_pow, **invest_params)[i],
        "$ \n",
        "Total investment cost:\t",
        "%.1f" % CAPEX_f(lifespan, **invest_params)[i],
        "$ \n",
        "Cost of H2 production:\t",
        "%.2f" % LCoH[i],
        "$/kg H2")

# Progress bar step
percent_complete = percent_complete + progress_step
progress_bar.progress(round(percent_complete, 3))

# '''
# ---- Calculating the Internal Rate of Return (IRR) and Net Present Value (NPV) ----
# '''

inflation = 0.02
discount_rate = 0.04

period_sz = 1  # year(s)
period_number = np.arange(0, lifespan + 1, period_sz)

yearly_mass_H2 = np.array(
    [sum(dataset_mass_H2[:, i]) * multiplier(data, period_sz)
     for i in range(len(dataset_mass_H2[0]))]
)  # kg

yearly_payment_mortgage = installment_pay(
    period_sz, interest_mortgage, share_mortgage, **invest_params)

yearly_payment_invest = installment_pay(
    period_sz, interest_invest, share_investment, **invest_params)

yearly_OPEX = (
    OPEX_f(data, choice_power_source, period_sz, dataset_input_pow, **invest_params)
    )

# Creating a list of the net cash flow for the years from t=0 to T=lifespan.

net_cf = [-CAPEX]
net_cf = net_cf + [
    (yearly_mass_H2 * H2_price
        + income_unused_pow(
            data,
            choice_power_source,
            Statement2,
            period_sz,
            dataset_input_pow,
            dataset_used_pow,
            **invest_params)  
        + income_FCR(Statement3, period_sz, **invest_params)
        - yearly_OPEX
        - yearly_payment_mortgage
        - yearly_payment_invest)
    * (1 + inflation) ** t
    for t in period_number[1:]
]

net_cf = np.array(net_cf)

IRR = np.array([npf.irr(net_cf[:, j]) for j in range(len(yearly_mass_H2))])

NPV = np.array([npf.npv(discount_rate, net_cf[:, j]) for j in range(len(yearly_mass_H2))])

# Progress bar step
percent_complete = percent_complete + progress_step
progress_bar.progress(round(percent_complete, 3))

# Calculating contributions to the positive cashflow for varying SOEC sizes over the 
# lifespan of the investment.
revenue_H2 = total_mass_H2 * H2_price

revenue_excess_heat = income_excess_heat(
    data, Statement1, lifespan, dataset_used_pow, **invest_params)

revenue_unused_pow = income_unused_pow(
    data, choice_power_source, Statement2, lifespan, dataset_input_pow, dataset_used_pow, **invest_params)

revenue_FCR = income_FCR(Statement3, period_sz, **invest_params)

end5 = time.time()
st.write("The entire calculation took %.3f second(s)." % (end5 - start1))

eval = pd.DataFrame({'SOEC size [kW]': sz_elec, 'LCoH [$/kg]': LCoH, 
        'IRR [%]': (IRR * 100), 'NPV [m $]': (NPV * 10**(-6))})

# '''
# ---------------- Plotting the development in different parameters ----------------
# '''

font = ""       # If a certain font is required, it can be defined here.

fig2, (ax1, ax2) = plt.subplots(2, sharex=True)

# Top subplot
color = "tab:green"
ax1.set_ylabel(r"LCoH [\$/kg H$_2$]", color=color, name=font)
ax1.plot([i / 1000 for i in sz_elec], LCoH, ".-", color=color)
ax1.grid()
ax1.tick_params(axis="y", labelcolor=color)
for tick in ax1.get_yticklabels():
    tick.set_fontname(font)

# Bottom subplot
color = "tab:red"  
ax2.set_xlabel("Size of SOEC plant [MW]", name=font)
ax2.set_ylabel(r"NPV [m \$]", color=color, name=font)
ax2.plot([i / 10 ** 3 for i in sz_elec], [i / 10 ** 6 for i in NPV], ".-", color=color)
ax2.grid()
ax2.tick_params(axis="y", labelcolor=color)
for tick in ax2.get_yticklabels():
    tick.set_fontname(font)

# Second axis that shares x-axis with bottom subplot
ax3 = ax2.twinx()
color = "tab:blue"
ax3.set_ylabel("IRR [%]", color=color, name=font)
ax3.plot([i / 1000 for i in sz_elec], [i * 100 for i in IRR], ".-", color=color)
ax3.tick_params(axis="y", labelcolor=color)
for tick in ax2.get_xticklabels():
    tick.set_fontname(font)
for tick in ax3.get_yticklabels():
    tick.set_fontname(font)

fig2.tight_layout()  # Otherwise the right y-label is slightly clipped
plt.show()
# fig.savefig(r'T:\01 - Hybrid Greentech ApS\07 - Research projects\06PF - NEXP2X\04_Assignments\Code\hydrogen-cost\hydrogen-cost_fig4.png', dpi=300)

st.subheader("Evaluation parameters for different SOEC sizes")
st.write(
    "For the assumptions specified below and the input given in the sidebar, the following LCoH, NPV, and IRR values are obtained:"
)
st.pyplot(fig2)

st.subheader("Values from calculation")
if st.checkbox("Show values from calculation"):
    st.write(eval)

# '''
# -------------------- Writing the assumptions in three columns --------------------
# '''
st.subheader("Assumptions")

col1, col2, col3 = st.columns(3)
col1.caption("General assumptions:")
col1.write(f"Dataset size = {dataset_size}")
col1.write("Investment lifespan = %.0f years" % lifespan)
col1.write(r"H$_2$ price = %.1f \$/kg" % H2_price)
col1.write("Power price = %.2f $/kWh" % pow_price)
col1.write("Excess heat price = %.6f $/kWh" % excess_heat_price)
col1.write("Conversions: 1 € = 1.15 $, and 1 DKK = 0.151 $")
col1.write(r"Excess heat is 5 % of the used power.")
col2.caption("SOEC assumptions:")
col2.write("Size: %.1f-%.1f MW" % (sz_elec_start / 1000, sz_elec_end / 1000))
col2.write("Stack cost = %.0f $/kW" % cost_stack)
col2.write("Stack lifetime = %.0f years" % stack_lifetime)
col2.write("BoP cost = %.0f $/kW" % cost_BoP)
col2.write("BoP lifetime = %.0f years" % BoP_lifetime)
col2.write("O&M = 0.02 \* (stack cost + BoP cost) \* size \* timeperiod")   # Change value here, if you change assumption later on.
col2.write(r"H$_2$ output factor = %.4f kg/kWh input" % H2_output_factor)
col3.caption("Wind farm assumptions:")
col3.write("Size: %.1f MW" % (sz_wind / 1000))
col3.write("Wind farm cost = %.1f $/kW" % cost_windfarm)
col3.write("Fixed O&M  = %.3f $/kW/year" % OnM_fixed)
col3.write("Variable O&M = %.7f $/kWh" % OnM_var)
col3.write("Technical lifetime = %.0f years" % windfarm_lifetime)

col4, col5, col6 = st.columns(3)
col4.caption("Assumptions related to IRR and NPV:")
col4.write("Period size = %.0f year" % (period_sz))
col4.write("Discount rate = %.2f %s/year" % (discount_rate * 100, "%"))
col4.write("Inflation = %.2f %s/year" % (inflation * 100, "%"))
col5.caption("Current limitations:")
col5.write("- Wind BoP is missing.")
col5.write("- Tariffs and equipment related to selling excess heat are presently not included.")
col5.write("- Assuming historical wind data and spot prices are representative of the future.")

# '''
# --------------------------------- Financial output section ---------------------------------
# '''
st.subheader("Financial output")

st.write(f"Overview of contributions to the positive cashflow for the {lifespan}-year period:")
fig3, ax1 = plt.subplots()

ax1.stackplot(
    (sz_elec / 10 ** 3),
    (revenue_H2 / 10 ** 6),
    (revenue_excess_heat / 10**6),
    (revenue_unused_pow / 10 ** 6),
    (revenue_FCR / 10 ** 6),
    labels=[r"Selling H$_2$", "Selling excess heat to district heating", "Selling unused power to grid", "Contributing to FCR"],
)
ax1.grid(alpha=0.3)
ax1.set_xlabel("Size of SOEC plant [MW]")
ax1.set_ylabel("Revenue [m $]")
ax1.legend(loc="lower right", bbox_to_anchor=(0.95, 0.01), prop={'size': 8})
st.pyplot(fig3)

st.write(f"Expenses over the {lifespan}-year period:")
financial_output = pd.DataFrame(
    {"SOEC size [kW]": sz_elec,
    "CAPEX [$]": CAPEX, 
    "OPEX [$]": OPEX,
    "Mortgage installment payment [$]": payment_mortgage,
    "Investor installment payment [$]": payment_invest,
    "Total expenses [$]": CAPEX + OPEX + payment_mortgage + payment_invest})

st.write(financial_output)

# ---------------------------------- Investment costs ----------------------------------
# ---------------- (bar plot showing what contributes to CAPEX and OPEX, ---------------
# --------------------- giving an idea of where money can be saved) --------------------
st.write(f"Overview of what contributes to the expenses for the {lifespan}-year period:")
CAPEX_contributions = CAPEX_equations(lifespan, **invest_params)[1]

total_SOEC_stack_cost = CAPEX_contributions[0]
total_SOEC_BoP_cost = CAPEX_contributions[1]
total_windfarm_cost = CAPEX_contributions[2]

OPEX_contributions = OPEX_equations(
    data, choice_power_source, lifespan, dataset_input_pow, **invest_params
    )[1]

OnM_SOEC = OPEX_contributions[0]
OnM_wind = OPEX_contributions[1] 
electricity_cost = OPEX_contributions[2]

labels = (
    ["Total wind farm cost", "O&M (wind farm)", "Total SOEC stack cost",   
    "Total SOEC BoP cost", "O&M (SOEC)", "Mortgage installment payment", 
    "Investor installment payment", "Total electricity cost"])

contributions = (np.array(
    [total_windfarm_cost, OnM_wind, total_SOEC_stack_cost,  
    total_SOEC_BoP_cost, OnM_SOEC, payment_mortgage, 
    payment_invest, electricity_cost]) / 10**6)

x_vals = sz_elec / 10**3
barwidth = sz_elec_step * 0.75 / 10**3

fig4, ax1 = plt.subplots()

ax1.bar(x_vals, contributions[0], width=barwidth, label=labels[0])
ax1.bar(x_vals, contributions[1], width=barwidth, bottom=contributions[0], label=labels[1])
ax1.bar(x_vals, contributions[2], width=barwidth, bottom=sum(contributions[0:2]), label=labels[2])
ax1.bar(x_vals, contributions[3], width=barwidth, bottom=sum(contributions[0:3]), label=labels[3])
ax1.bar(x_vals, contributions[4], width=barwidth, bottom=sum(contributions[0:4]), label=labels[4])
ax1.bar(x_vals, contributions[5], width=barwidth, bottom=sum(contributions[0:5]), label=labels[5])
ax1.bar(x_vals, contributions[6], width=barwidth, bottom=sum(contributions[0:6]), label=labels[6])
ax1.bar(x_vals, contributions[7], width=barwidth, bottom=sum(contributions[0:7]), label=labels[7])

ax1.legend(loc="lower left", bbox_to_anchor=(0.05, 0.01), prop={'size': 8})
ax1.set_axisbelow(True)
ax1.minorticks_on()                 # In order to have minor grid lines on.
ax1.grid(which='both', alpha=0.5)
ax1.set_xlabel("Size of SOEC plant [MW]")
ax1.set_ylabel("Cost [m $]")

st.pyplot(fig4)

st.write("The CAPEX is determined by: The total wind farm cost, total SOEC stack cost, and total SOEC BoP cost.")
st.write("The OPEX is determined by: The O&M for both wind farm and SOEC, and the total electricity cost.")

# '''
# ---------------------- Section with power-to-fuel calculations ----------------------
# '''
st.subheader("Power-to-fuel")
if st.checkbox("Estimate LCo-methanol?"):
    st.write("Equation used for the (rough) estimation:")
    st.write(r"$LCoM = LHV_{MeOH} \times \frac{LCoH}{LHV_{H_2}}$")

    LCoM = LHV_MeOH * LCoH / LHV_H2
    
    eval_methanol = pd.DataFrame({'SOEC size [kW]': sz_elec, 'LCoM [$/kg]': LCoM})
    
    fig5, ax1 = plt.subplots()
    color = "tab:orange"
    ax1.set_ylabel(r"LCoM [\$/kg MeOH]", color=color, name=font)
    ax1.plot([i / 1000 for i in sz_elec], LCoM, ".-", color=color)
    ax1.grid()
    ax1.tick_params(axis="y", labelcolor=color)
    for tick in ax1.get_yticklabels():
        tick.set_fontname(font)

    st.pyplot(fig5)

    st.write(eval_methanol)