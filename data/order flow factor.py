# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:32:49 2022

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

tickers = ["EVO","SINCH","LATO_B","KINV_B","NIBE_B","EQT","MIPS","STORY_B","SF","PDX","SBB_B","BALD_B","SAGA_B","INDT","LIFCO_B","LAGR_B"]

stock_returns = {}

for x in tickers:
    print(x)
    data = pd.read_csv(str(x) + '_20220128.csv',sep=";",header=None)
    
    buyer = data.iloc[:,11] == ">"
    seller = data.iloc[:,11] == "<"
    signed_volume = buyer*data.iloc[:,4] - seller*data.iloc[:,4]
    order_flow = signed_volume.sum()
    total_volume = data.iloc[:,4].sum()
    print(str(x) + " Order flow " + str(order_flow))
    print("Percent of volume " + str(order_flow/total_volume))



#data[data.iloc[:,11] == ">",11] = 1
#data[data.iloc[:,11] == "<",11] = -1

#data = pd.DataFrame(data)