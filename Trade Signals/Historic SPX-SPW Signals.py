# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:05:58 2021

@author: MI2
"""
import pandas as pd
import tia.bbg.datamgr as dm
from datetime import date as dt
import matplotlib.pyplot as plt

mgr = dm.BbgDataManager() 

[win1,win2,win3] = 5,20,50 #can only use windows of 5,10,20,30,40,50,60,100,120,180,200 or you need excel input

ticker1 = ['SPX Index']
ticker2 = ['SPW Index']
fields = ['PX_LAST', 'MOV_AVG_'+str(win1)+'D', 'MOV_AVG_'+str(win2)+'D', 'MOV_AVG_'+str(win3)+'D']
dataPeriod = "WEEKLY" #Anything below is significantly cramped
startDate = dt(1972,1,1)
endDate = dt.today()

sids1= mgr[ticker1]
df1 = sids1.get_historical(fields,startDate,endDate,dataPeriod)
df1 = df1.dropna()
df1.columns = df1.columns.droplevel(0)

dates1 = df1.index.tolist()

xwin1= df1[fields[1]].tolist()
xwin2 = df1[fields[2]].tolist()
xwin3 = df1[fields[3]].tolist()

sids2 = mgr[ticker2]
df2 = sids2.get_historical(fields,startDate,endDate,dataPeriod)
df2 = df2.dropna()
df2.columns = df2.columns.droplevel(0)

dates2 = df2.index.tolist()

wwin1 = df2[fields[1]].tolist()
wwin2 = df2[fields[2]].tolist()
wwin3 = df2[fields[3]].tolist()

buySignalIndexPrelim1 = []
sellSignalIndexPrelim1 = []

for i in range(1,len(dates1)):
    if xwin1[i] > xwin2[i] and xwin2[i] > xwin3[i] and xwin2[i-1] < xwin3[i-1]:
        buySignalIndexPrelim1.append(i)
        
for i in range(1,len(dates1)):
    if xwin1[i] < xwin2[i] and xwin2[i] < xwin3[i] and xwin2[i-1] > xwin3[i-1]:
        sellSignalIndexPrelim1.append(i)

buySignalIndexPrelim2 = []
sellSignalIndexPrelim2 = []

for i in range(1,len(dates2)):
    if wwin1[i] > wwin2[i] and wwin2[i] > wwin3[i] and wwin2[i-1] < wwin3[i-1]:
        buySignalIndexPrelim2.append(i)
        
for i in range(1,len(dates2)):
    if wwin1[i] < wwin2[i] and wwin2[i] < wwin3[i] and wwin2[i-1] > wwin3[i-1]:
        sellSignalIndexPrelim2.append(i)

buySignalIndex1 = []
sellSignalIndex1 =[]
for k in buySignalIndexPrelim1:
    s = []
    for sig in sellSignalIndexPrelim1:
        if sig > k:
            s.append(sig)
    if len(s) > 1:
        if s[0] not in sellSignalIndex1:
            buySignalIndex1.append(k)
            sellSignalIndex1.append(s[0])
            print(k,s[0])
            print(dates1[k],dates1[s[0]])
        else:
            print('Duplicate!',k,s[0])
    elif k > len(sellSignalIndexPrelim1):
        buySignalIndex1.append(k)
    else:
        if len(s) == 1:
            if s[0] not in sellSignalIndex1:
                buySignalIndex1.append(k)
                sellSignalIndex1.append(s[0])
                print(k,s[0])
                print(dates1[k],dates1[s[0]])
            else:
                print('Duplicate!',k,s[0])
        else:
            break
   
buySignalIndex2 = []
sellSignalIndex2 =[]
for k in buySignalIndexPrelim2:
    s = []
    for sig in sellSignalIndexPrelim2:
        if sig > k:
            s.append(sig)
    if len(s) > 1:
        if s[0] not in sellSignalIndex2:
            buySignalIndex2.append(k)
            sellSignalIndex2.append(s[0])
            print(k,s[0])
            print(dates2[k],dates2[s[0]])
        else:
            print('Duplicate!',k,s[0])
    else:
        if len(s) == 1:
            if s[0] not in sellSignalIndex2:
                buySignalIndex2.append(k)
                sellSignalIndex2.append(s[0])
                print(k,s[0])
                print(dates2[k],dates2[s[0]])
            else:
                print('Duplicate!',k,s[0])
        else:
            break

#APPENDS END BUY SIGNALS IF NOT IN PAIR WITH SELL SIGNAL
if buySignalIndex1[-1] != buySignalIndexPrelim1[-1]:
    add = buySignalIndexPrelim1[-1]
    buySignalIndex1.append(add)
    print('Adding final SPX buy signal',dates1[add])
else:
    print('No new SPX Buy additions!')
    
if buySignalIndex2[-1] != buySignalIndexPrelim2[-1]:
    add = buySignalIndexPrelim2[-1]
    buySignalIndex2.append(add)
    print('Adding final SPW buy signal',dates2[add])
else:
    print('No new SPW Buy additions!')
    
if sellSignalIndex1[-1] != sellSignalIndexPrelim1[-1]:#
    add = sellSignalIndexPrelim1[-1]
    print(add)
    sellSignalIndex1.append(add)
    print('Adding final SPX sell signal',dates1[add])
else:
    print('No new SPX Sell additions!')
    
if sellSignalIndex2[-1] != sellSignalIndexPrelim2[-1]:
    add = sellSignalIndexPrelim2[-1]
    sellSignalIndex2.append(add)
    print('Adding final SPW sell signal',dates2[add])
else:
    print('No new SPW Sell additions!')


print('----------------------------------------------SPX SIGNAL DATES----------------------------------------')
SPX_Buy = []
SPX_Sell = []
for i in range(0,len(buySignalIndex1)):
    SPX_Buy.append(dates1[buySignalIndex1[i]])

for i in range(0,len(sellSignalIndex1)):
    SPX_Sell.append(dates1[sellSignalIndex1[i]])

print('BUY Signals: ',SPX_Buy)
print('                     ')
print('SELL Signals: ',SPX_Sell)

print('---------------------------------------------SPW SIGNAL DATES------------------------------------------')
SPW_Buy = []
SPW_Sell = []
for i in range(0,len(buySignalIndex2)):
    SPW_Buy.append(dates2[buySignalIndex2[i]])

for i in range(0,len(sellSignalIndex2)):
    SPW_Sell.append(dates2[sellSignalIndex2[i]])

print('BUY Signals: ',SPW_Buy)
print('                      ')
print('SELL Signals: ',SPW_Sell)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.title.set_text('SPX Index Signals')
ax1.plot(df1[fields[0]], label=fields[0])
ax1.plot(df1[fields[1]], label=fields[1])
ax1.plot(df1[fields[2]], label=fields[2])
ax1.plot(df1[fields[3]], label=fields[3])
for i in range(0,len(SPX_Sell)):
    ax1.axvline(x = SPX_Sell[i], color = 'r')
for i in range(0,len(SPX_Buy)):
    ax1.axvline(x = SPX_Buy[i], color = 'g')
ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
ax1.grid()

ax2.title.set_text('SPW Index Signals')
ax2.plot(df2[fields[0]], label=fields[0])
ax2.plot(df2[fields[1]], label=fields[1])
ax2.plot(df2[fields[2]], label=fields[2])
ax2.plot(df2[fields[3]], label=fields[3])
for i in range(0,len(SPW_Sell)):
    ax2.axvline(x = SPW_Sell[i], color = 'r')
for i in range(0,len(SPW_Buy)):
    ax2.axvline(x = SPW_Buy[i], color = 'g')
ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
ax2.grid()
plt.show()