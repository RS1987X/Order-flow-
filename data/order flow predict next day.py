# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:03:05 2022

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
percent_vol_data = []

dates = (["20220128","20220131","20220201","20220202","20220203","20220204","20220207",
         "20220208","20220209","20220210","20220211","20220214","20220215","20220216", 
         "20220217"])

for d in dates:
    #print(" ")
    #print(d)
    #print(" ")
    for x in tickers:
        
        #print(x)
        data = pd.read_csv(x + '_' + d +'.csv',sep=";",header=None)
        
        buyer = (data.iloc[:,11] == ">") & (data.iloc[:,7] != "CROS")
        seller = (data.iloc[:,11] == "<") & (data.iloc[:,7] != "CROS")
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
        not_opening_closing_call = data.iloc[:,11].notna()
        opening_period = data.iloc[:,1] <= "17:30:00"
        opening_of = (buyer*data.loc[opening_period,4] - seller*data.loc[opening_period,4]).sum()
        opening_tot_volume = data.loc[opening_period & not_opening_closing_call,4].sum()
        
        percent_of_opening_vol = opening_of/opening_tot_volume
        percent_of_opening_vol_str = "{:.1%}".format(percent_of_opening_vol)
        print("Opening OF " + str(opening_of))
        print("OF % opening volume " + percent_of_opening_vol_str) 
        
        #opening return
        opening_period_prices = data.loc[opening_period & not_opening_closing_call,2]
        opening_period_return = float(opening_period_prices.iloc[0].replace(",","."))/float(opening_period_prices.tail(1).iloc[0].replace(",","."))-1
        opening_period_low = float(opening_period_prices.min().replace(",","."))
        opening_period_high = float(opening_period_prices.max().replace(",","."))

        #return from vwap 9:20-9:22 to 11:00:00
        close_price = data.loc[(data.iloc[:,1] > "17:20:00") & (data.iloc[:,1] < "17:30:00"),2]
        #close_price = float(data.iloc[0,2].replace(",","."))
        close_price = close_price.apply(lambda x: x.replace(',','.')).astype(float)
        close_price = close_price.mean()
      
        of_data.append((x,d,percent_of_opening_vol,close_price))


#percent of total on bid/ask demeaned & normalized
of_data = np.array(of_data)
of = of_data[of_data[:,0]=="EVO",2]
of = of[:, np.newaxis]
close = of_data[of_data[:,0]=="EVO",3]
close = close[:, np.newaxis]

for x in tickers[1:]:
    of_to_append = of_data[of_data[:,0]==x,2][:,np.newaxis]
    of = np.append(of,of_to_append, axis=1)
    close_pr_to_append =  of_data[of_data[:,0]==x,3][:,np.newaxis]
    close =  np.append(close,close_pr_to_append, axis=1)
    
of = of.astype(float)
of_df = pd.DataFrame(of,columns=tickers)
of_pos = of_df[of_df>=0]
of_neg = of_df[of_df<0]

of_pos_mean = of_pos.expanding(min_periods=3).mean()[~of_pos.isna()]
of_pos_std = of_pos.expanding(min_periods=3).std()[~of_pos.isna()]
of_pos_n_std_dev = (of_pos-of_pos_mean.shift(1))/of_pos_std.shift(1)
of_neg_mean = of_neg.expanding(min_periods=3).mean()[~of_neg.isna()]
of_neg_std = of_neg.expanding(min_periods=3).std()[~of_neg.isna()]
of_neg_n_std_dev = (of_neg-of_neg_mean.shift(1))/of_neg_std.shift(1)

close = close.astype(float)
close_df  = pd.DataFrame(close,columns=tickers)
returns_df = close_df.pct_change()

returns_of_pos = returns_df[~of_pos_n_std_dev.isna()]
returns_of_neg = returns_df[~of_neg_n_std_dev.isna()]

plt.figure(3)
plt.scatter(of_df,returns_df)
lm = sm.OLS(returns_df.stack(),of_df.iloc[1:,:].stack(),missing='drop').fit()
print(lm.summary())
