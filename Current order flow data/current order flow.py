# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:19:56 2022

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
of_data = []
for x in tickers:
    print(x)
    data = pd.read_csv(x + '.csv',sep=";",header=None)
    
    buyer = (data.iloc[:,11] == ">") & (data.iloc[:,7] != "CROS")
    seller = (data.iloc[:,11] == "<") & (data.iloc[:,7] != "CROS")
    tot_buy_volume = buyer*data.iloc[:,4]
    tot_sell_volume = seller*data.iloc[:,4]
    signed_volume = tot_buy_volume - tot_sell_volume
    order_flow = signed_volume.sum()
    not_opening_closing_call = data.iloc[:,11].notna()
    total_volume = data.loc[not_opening_closing_call,4].sum()
    #print("Order flow (OF) " + str(order_flow))
    #percent_of_vol = order_flow/total_volume
    #percent_of_vol = "{:.1%}".format(percent_of_vol)
    #print("OF % of volume " + str(percent_of_vol))
    
    #opening order flow (first 15 minutes)
    not_opening_closing_call = data.iloc[:,11].notna()
    opening_period = data.iloc[:,1] < "09:15:00"
    opening_of = (buyer*data.loc[opening_period,4] - seller*data.loc[opening_period,4]).sum()
    opening_tot_volume = data.loc[opening_period & not_opening_closing_call,4].sum()
    percent_of_opening_vol = opening_of/opening_tot_volume
    percent_of_opening_vol_str = "{:.1%}".format(percent_of_opening_vol)
    print("Opening OF " + str(opening_of))
    print("OF % opening volume " + percent_of_opening_vol_str)
    print(" ")
    of_data.append((x,percent_of_opening_vol,opening_of))

of_data.sort(key = lambda x: x[1])
