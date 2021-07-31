# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:01:43 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from pandas.tseries.offsets import BDay

startDate = dt(1980,1,1)
endDate = dt(2020,12,31)
dataPeriod = 'DAILY'

ticker = 'BA US Equity'

mgr = dm.BbgDataManager() 

sids = mgr[ticker]

VWAP = sids.get_historical(['EQY_WEIGHTED_AVG_PX', 'VWAP_VOLUME'],startDate,endDate, dataPeriod)

VWAP = VWAP.dropna()

lengthOfRoll = 100
lengthOfRoll2 = 55

VWAPProd = []
for i in range(0,len(VWAP.index)):
    VWAPProd.append(VWAP['EQY_WEIGHTED_AVG_PX'][i]*VWAP['VWAP_VOLUME'][i])

rollingVWAP = np.ones([len(VWAP.index)-lengthOfRoll,1]) #rolling avg
for i in range(0,len(VWAP.index ) - lengthOfRoll):
   rollingVWAP[i] = sum(VWAPProd[i:i+lengthOfRoll-1])/sum(VWAP['VWAP_VOLUME'][i:i+lengthOfRoll-1])

rollingVWAP2 = np.ones([len(VWAP.index)-lengthOfRoll2,1])
for i in range(0,len(VWAP.index) - lengthOfRoll2):
   rollingVWAP2[i] = sum(VWAPProd[i:i+lengthOfRoll2-1])/sum(VWAP['VWAP_VOLUME'][i:i+lengthOfRoll2-1])

VWAPStart = dt(2000,3,14)
VWAPStart = VWAPStart + 0*BDay()

aggVWAP = []
for i in range(0,len(VWAP.index)):
    if VWAP.index[i] == VWAPStart:
        point = i

for i in range(point+1,len(VWAP.index)):
    VWAP_Ag = sum(VWAPProd[point:i])/sum(VWAP['VWAP_VOLUME'][point:i])
    aggVWAP.append(VWAP_Ag)

aggVWAP = pd.DataFrame(data=aggVWAP, index=VWAP.index[point:-1])

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title(ticker + " " + str(lengthOfRoll) + " " + str(lengthOfRoll2))
plt.plot(np.array(VWAP['EQY_WEIGHTED_AVG_PX'][lengthOfRoll+1:]))
plt.plot(rollingVWAP)
plt.plot(rollingVWAP2[lengthOfRoll-lengthOfRoll2+1:])
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('Aggregate Averages')
plt.plot(VWAP['EQY_WEIGHTED_AVG_PX'])
plt.plot(aggVWAP)
plt.show()