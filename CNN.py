# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:37:38 2021

@author: MI2
"""
import numpy as np
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from Norm import *

startDate = dt(1999,1,1)
endDate = dt(2020,12,31)
today = dt.today()

mgr = dm.BbgDataManager() 

vbs = ['MOODCBAA Index','USGG10YR Index', 'VIX Index', 'SPX Index','TLT US Equity', 'SPY US Equity', 'PCUSEQTR Index','NWHLNYHI Index','NWHLNYLO Index','SUM INX Index']

sids = mgr[vbs]
sids.PX_LAST
df1 = sids.get_historical(['PX_Last'],startDate,endDate)

df1 = df1.fillna(method = 'ffill')
df1.columns = df1.columns.droplevel(1)

print(df1)

sp = df1['MOODCBAA Index'] - df1['USGG10YR Index']
vix = normal_cal(df1['VIX Index']) * 100
vix50 = vix.rolling(window = 50).mean()
spx = normal_cal(df1['SPX Index']) * 100
spx125 = spx.rolling(window = 125).mean()
spxratio = spx/spx125
diff1 = df1['SPY US Equity'] / df1['TLT US Equity']
diff = normal_cal(diff1) * 100
pc = normal_cal(df1['PCUSEQTR Index']) * 100
pc125 = pc.rolling(window = 125).mean()
idd = normal_cal((df1['NWHLNYHI Index'] - df1['NWHLNYLO Index'])) * 100
idd52 = (normal_cal(idd)*100).rolling(window = 52).mean() #double normalized, never used???
summ = normal_cal(df1['SUM INX Index']) * 100

spread = normal_cal(sp) * 100
finValue = ((1/7 * spread) + (1/7 * vix50) + (1/7 * spxratio) + (1/7 * diff) + (1/7 * pc125) + (1/7 * idd) + (1/7 * summ))

print(finValue)

#Plot figure
plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(finValue,label='FG Index')
plt.legend(loc=1)

finValue.to_excel(r'C:\Users\MI2\Desktop\Excel\CNN_FearGreed_Export.xlsx', index = True) #exports to excel