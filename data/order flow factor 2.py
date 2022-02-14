# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:46:50 2022

@author: richa
"""

#order flow estimation is done by labeling transaction as buy if transaction is done on offer, sell if done on bid
#we are thus not able to label the auctions or the mid spread transactions.
#Further we can compare by labeling transactions as buy or sell if price tick is up or down
#possibly also create variable price response which is the amount of change (up or down) for a given amount of volume, 
#look at 3 or 5 minute bins

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
import os

tickers = ["EVO","SINCH","LATO_B","KINV_B","NIBE_B","EQT","MIPS","STORY_B","SF","PDX","SBB_B","BALD_B","SAGA_B","INDT","LIFCO_B","LAGR_B"]

of_data = []
op_dislocation_data = []
i = 0
directory = os.fsencode("C:\Users\richa\Documents\GitHub\Order-flow-\data")

#,"20220131","20220201" #"20220128"
#"20220128","20220131","20220201","20220202","20220203","20220204","20220207","20220208","20220209","20220210"
dates = ["20220204","20220207","20220208","20220209","20220210","20220211"]
for d in dates:
    print(" ")
    print(d)
    print(" ")
    for x in tickers:
        
        print(x)
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
        opening_period = data.iloc[:,1] < "09:15:00"
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
# =============================================================================
#         #calculate fair market impact
#         sigma = 0.035
#         Y = 0.7
#         opening_buy_volume = tot_buy_volume.loc[opening_period & not_opening_closing_call].sum()
#         opening_sell_volume = tot_sell_volume.loc[opening_period & not_opening_closing_call].sum()
#         theor_buy_mi = Y*sigma*math.sqrt(opening_buy_volume/opening_tot_volume)
#         theor_sell_mi = Y*sigma*math.sqrt(opening_sell_volume/opening_tot_volume)
#         theor_mi = theor_buy_mi-theor_sell_mi
#         
#         #judge overeaction by comparing opening return with estimate of fair market impact
#         opening_dislocation = opening_period_return - theor_mi
#         op_dislocation_str = "{:.1%}".format(opening_dislocation)
#         print("Dislocation (+overbought/-oversold) " + op_dislocation_str)
#
# =============================================================================
        #return from vwap 9:20-9:22 to 11:00:00
        close_price = data.loc[(data.iloc[:,1] > "10:50:00") & (data.iloc[:,1] < "10:59:00"),2]
        #close_price = float(data.iloc[0,2].replace(",","."))
        close_price = close_price.apply(lambda x: x.replace(',','.')).astype(float)
        close_price = close_price.mean()
        
# =============================================================================
#         #calculate if stop loss triggered
#         prices_post_entry = data.loc[(data.iloc[:,1] > "09:22:00") & (data.iloc[:,1] < "17:20:00"),2]
#         
#         if order_flow > 0:
#             stop_level = float(opening_period_prices.min().replace(",","."))
#             
#             if float(prices_post_entry.min().replace(",",".")) <= stop_level:
#                 close_price = stop_level
#         else:
#             stop_level = float(opening_period_prices.max().replace(",","."))
#             if float(prices_post_entry.max().replace(",",".")) >= stop_level:
#                 close_price = stop_level
# =============================================================================
        
# =============================================================================
#         #enter trade on retracement
#         prices_trade_window = data.loc[(data.iloc[:,1] > "09:22:00") & (data.iloc[:,1] < "10:50:00"),2]
#         prices_trade_window = prices_trade_window.apply(lambda x: x.replace(',','.')).astype(float)
#         volumes_trade_window = data.loc[(data.iloc[:,1] > "09:22:00") & (data.iloc[:,1] < "10:50:00"),4]
#         vwap_trade_window = np.cumsum(prices_trade_window*volumes_trade_window)/volumes_trade_window.cumsum()
#         if order_flow > 0:
#             idx = (prices_trade_window <= vwap_trade_window)[::-1].idxmax()
#         else:
#             idx = (prices_trade_window >= vwap_trade_window)[::-1].idxmax()
#         
#         entry_price = vwap_trade_window[idx]
#         
# =============================================================================
        #entry price
        entry_period_volumes = data.loc[(data.iloc[:,1] > "09:20:00") & (data.iloc[:,1] < "09:23:00"),4]
        entry_period_prices = data.loc[(data.iloc[:,1] > "09:20:00") & (data.iloc[:,1] < "09:23:00"),2]
        entry_period_prices = entry_period_prices.apply(lambda x: x.replace(',','.')).astype(float)
        entry_price = np.multiply(entry_period_prices,entry_period_volumes).sum() / entry_period_volumes.sum()
        

        #calculate return
        ret = close_price/entry_price-1
        ret_str = "{:.1%}".format(ret)
        print("Post open return to close " + ret_str)
        print(" ")
        of_data.append((x,percent_of_opening_vol,ret))
        #op_dislocation_data.append((x,opening_dislocation,ret))
        i = i + 1

percent_of_opening_vols = [x[1] for x in of_data]
returns = [x[2] for x in of_data]

of_data.sort(key = lambda x: x[1])
of_data_df = pd.DataFrame(of_data, columns=['name', 'percent_of_opening_vol', 'returns'])
threshold = 0.3
stocks_w_signal = of_data_df[(of_data_df["percent_of_opening_vol"] < -threshold) | (of_data_df["percent_of_opening_vol"] > threshold)]

shorts = stocks_w_signal["percent_of_opening_vol"] < 0
longs = stocks_w_signal["percent_of_opening_vol"] > 0
strat_ret = longs*stocks_w_signal["returns"] - shorts*stocks_w_signal["returns"]
print("Avg trade return " + str(strat_ret.mean()))
print("Volatility of returns " + str(strat_ret.std()))
print("Kelly f " + str(strat_ret.mean()/strat_ret.std()**2))
print(" ")

# =============================================================================
# 
# dislocation_of_stock = [x[1] for x in op_dislocation_data]
# return_of_stock = [x[2] for x in op_dislocation_data]
# # 
# op_dislocation_data.sort(key = lambda x: x[1])
# =============================================================================
# print(" ")
# #longs
# print("LONGS")
# print(str(op_dislocation_data[0]))
# print(str(op_dislocation_data[1]))
# #shorts
# print("SHORTS")
# print(str(op_dislocation_data[-1]))
# print(str(op_dislocation_data[-2]))



plt.figure(1)
plt.scatter(percent_of_opening_vols,returns)
lm = sm.OLS(returns,percent_of_opening_vols,missing='drop').fit()
print(lm.summary())

plt.figure(2)
plt.scatter(stocks_w_signal["percent_of_opening_vol"],stocks_w_signal["returns"])
lm2 = sm.OLS(stocks_w_signal["returns"],stocks_w_signal["percent_of_opening_vol"],missing='drop').fit()
print(lm2.summary())



# =============================================================================
# 
# 
# #sm.add_constant(
# plt.figure(3)
# plt.scatter(dislocation_of_stock,return_of_stock)
# lm3 = sm.OLS(return_of_stock,dislocation_of_stock,missing='drop').fit()
# print(lm3.summary())
# 
# 
# =============================================================================






#data[data.iloc[:,11] == ">",11] = 1
#data[data.iloc[:,11] == "<",11] = -1

#data = pd.DataFrame(data)