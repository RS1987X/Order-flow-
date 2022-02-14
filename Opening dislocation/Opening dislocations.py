# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 00:06:41 2022

@author: richa
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import date
from datetime import datetime
from dateutil import parser
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

tickers = ["EVO","SINCH","LATO_B","KINV_B","NIBE_B","EQT","MIPS","STORY_B","SF","PDX","SBB_B","BALD_B","SAGA_B","INDT","LIFCO_B","LAGR_B"]


op_dislocation_data = []

for x in tickers:
    print(x)
    data = pd.read_csv(x +'.csv',sep=";",header=None)
    
    buyer = data.iloc[:,11] == ">"
    seller = data.iloc[:,11] == "<"
    tot_buy_volume = buyer*data.iloc[:,4]
    tot_sell_volume = seller*data.iloc[:,4]
    signed_volume = tot_buy_volume - tot_sell_volume
    order_flow = signed_volume.sum()
    not_opening_closing_call = data.iloc[:,11].notna()
    total_volume = data.loc[not_opening_closing_call,4].sum()
    print("Order flow (OF) " + str(order_flow))
    percent_of_vol = order_flow/total_volume
    percent_of_vol = "{:.1%}".format(percent_of_vol)
    print("OF % of volume " + str(percent_of_vol))

    #opening order flow (first 15 minutes)
    opening_of = (buyer*data.loc[data.iloc[:,1] <= "09:15:00",4] - seller*data.loc[data.iloc[:,1] <= "09:15:00",4]).sum()
    not_opening_closing_call = data.iloc[:,11].notna()
    opening_period = data.iloc[:,1] <= "09:15:00"
    opening_tot_volume = data.loc[opening_period & not_opening_closing_call,4].sum()
    
    percent_of_opening_vol = opening_of/opening_tot_volume
    percent_of_opening_vol_str = "{:.1%}".format(percent_of_opening_vol)
    print("OF % opening volume " + percent_of_opening_vol_str)
    
    #opening return
    opening_period_prices = data.loc[opening_period & not_opening_closing_call,2]
    opening_period_return = float(opening_period_prices.iloc[0].replace(",","."))/float(opening_period_prices.tail(1).iloc[0].replace(",","."))-1
    
    #calculate fair market impact
    sigma = 0.02
    Y = 0.7
    opening_buy_volume = tot_buy_volume.loc[opening_period & not_opening_closing_call].sum()
    opening_sell_volume = tot_sell_volume.loc[opening_period & not_opening_closing_call].sum()
    theor_buy_mi = Y*sigma*math.sqrt(opening_buy_volume/opening_tot_volume)
    theor_sell_mi = Y*sigma*math.sqrt(opening_sell_volume/opening_tot_volume)
    theor_mi = theor_buy_mi-theor_sell_mi
    
    #judge overeaction by comparing opening return with estimate of fair market impact
    opening_dislocation = opening_period_return - theor_mi
    op_dislocation_str = "{:.1%}".format(opening_dislocation)
    print("Dislocation (+overbought/-oversold) " + op_dislocation_str)
   
    op_dislocation_data.append((x,opening_dislocation))


dislocation_of_stock = [x[1] for x in op_dislocation_data]
return_of_stock = [x[2] for x in op_dislocation_data]

op_dislocation_data.sort(key = lambda x: x[1])

#shorts
print("SHORTS")
print(str(op_dislocation_data[-1]))
print(str(op_dislocation_data[-2]))

#longs
print("LONGS")
print(str(op_dislocation_data[0]))
print(str(op_dislocation_data[1]))