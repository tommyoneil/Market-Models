# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:00:47 2021

@author: MI2
"""
import pandas as pd
import tia.bbg.datamgr as dm
from datetime import date as dt
import matplotlib.pyplot as plt

mgr = dm.BbgDataManager() 

[win1,win2,win3] = 5,20,50

ticker1 = ['SPX Index']
ticker2 = ['SPW Index']
fields = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'MOV_AVG_'+str(win1)+'D', 'MOV_AVG_'+str(win2)+'D', 'MOV_AVG_'+str(win3)+'D']
dataPeriod = "WEEKLY"
startDate = dt(1972,1,1)
endDate = dt.today()

sids1= mgr[ticker1]
df1 = sids1.get_historical(fields,startDate,endDate,dataPeriod)
df1 = df1.dropna()
df1.columns = df1.columns.droplevel(0)

dates1 = df1.index.tolist()
opens1 = df1['PX_OPEN'].tolist()
highs1 = df1['PX_HIGH'].tolist()
closes1 = df1['PX_LAST'].tolist()

xwin1= df1[fields[4]].tolist()
xwin2 = df1[fields[5]].tolist()
xwin3 = df1[fields[6]].tolist()

sids2 = mgr[ticker2]
df2 = sids2.get_historical(fields,startDate,endDate,dataPeriod)
df2 = df2.dropna()
df2.columns = df2.columns.droplevel(0)

dates2 = df2.index.tolist()
opens2 = df2['PX_OPEN'].tolist()
highs2 = df2['PX_HIGH'].tolist()
closes2 = df2['PX_LAST'].tolist()

wwin1 = df2[fields[4]].tolist()
wwin2 = df2[fields[5]].tolist()
wwin3 = df2[fields[6]].tolist()

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

maxCloseDateIndex1 = []
maxHighDateIndex1 = []
signalMaxCloses1 = []
signalMaxHighs1 = []
for i in range(len(buySignalIndex1)):
    try:
        sigMaxC1 = max(closes1[buySignalIndex1[i]:sellSignalIndex1[i]])
        sigMaxH1 = max(highs1[buySignalIndex1[i]:sellSignalIndex1[i]])
        sigMaxCD1 = [ix for ix in range(len(closes1)) if closes1[ix] == sigMaxC1 and ix <= sellSignalIndex1[i] 
                    and ix >= buySignalIndex1[i]][0]
        sigMaxHD1 = highs1.index(sigMaxH1)
        
        signalMaxCloses1.append(sigMaxC1)
        signalMaxHighs1.append(sigMaxH1)
        maxCloseDateIndex1.append(sigMaxCD1)
        maxHighDateIndex1.append(sigMaxHD1)
    except: #Last signal is until the end of series
        sigMaxC1 = max(closes1[buySignalIndex1[i]:])
        sigMaxH1 = max(highs1[buySignalIndex1[i]:])
        sigMaxCD1 = closes1.index(sigMaxC1)
        sigMaxHD1 = highs1.index(sigMaxH1)
        
        signalMaxCloses1.append(sigMaxC1)
        signalMaxHighs1.append(sigMaxH1)
        maxCloseDateIndex1.append(sigMaxCD1)
        maxHighDateIndex1.append(sigMaxHD1)

maxCloseDateIndex2 = []
maxHighDateIndex2 = []
signalMaxCloses2 = []
signalMaxHighs2 = []
for i in range(len(buySignalIndex2)):
    try:
        sigMaxC2 = max(closes2[buySignalIndex2[i]:sellSignalIndex2[i]])
        sigMaxH2 = max(highs2[buySignalIndex2[i]:sellSignalIndex2[i]])
        sigMaxCD2 = [ix for ix in range(len(closes2)) if closes2[ix] == sigMaxC2 and ix <= sellSignalIndex2[i] 
                    and ix >= buySignalIndex2[i]][0]
        #sigMaxCD = closes.index(sigMaxC)
        sigMaxHD2 = highs2.index(sigMaxH2)
        
        signalMaxCloses2.append(sigMaxC2)
        signalMaxHighs2.append(sigMaxH2)
        maxCloseDateIndex2.append(sigMaxCD2)
        maxHighDateIndex2.append(sigMaxHD2)
    except: #Last signal is until the end of series
        sigMaxC2 = max(closes2[buySignalIndex2[i]:])
        sigMaxH2 = max(highs2[buySignalIndex2[i]:])
        sigMaxCD2 = closes2.index(sigMaxC2)
        sigMaxHD2 = highs2.index(sigMaxH2)
        
        signalMaxCloses2.append(sigMaxC2)
        signalMaxHighs2.append(sigMaxH2)
        maxCloseDateIndex2.append(sigMaxCD2)
        maxHighDateIndex2.append(sigMaxHD2)

returnsCloses1 = []
returnsHighs1 = []
for i in range(len(buySignalIndex1)):
    retC1 = (signalMaxCloses1[i] - closes1[buySignalIndex1[i]])/ closes1[buySignalIndex1[i]] * 100
    retH1 = (signalMaxHighs1[i] - closes1[buySignalIndex1[i]])/ closes1[buySignalIndex1[i]] * 100
    returnsCloses1.append(retC1)
    returnsHighs1.append(retH1)
    
returnsCloses2 = []
returnsHighs2 = []
for i in range(len(buySignalIndex2)):
    retC2 = (signalMaxCloses2[i] - closes2[buySignalIndex2[i]])/ closes2[buySignalIndex2[i]] * 100
    retH2 = (signalMaxHighs2[i] - closes2[buySignalIndex2[i]])/ closes2[buySignalIndex2[i]] * 100
    returnsCloses2.append(retC2)
    returnsHighs2.append(retH2)
    
maxCloseDates1 = [dates1[k] for k in maxCloseDateIndex1]
maxHighDates1 = [dates1[k] for k in maxHighDateIndex1]
signalDates1 = [dates1[k] for k in buySignalIndex1]
signalCloses1 = [closes1[k] for k in buySignalIndex1]

maxCloseDates2 = [dates2[k] for k in maxCloseDateIndex2]
maxHighDates2 = [dates2[k] for k in maxHighDateIndex2]
signalDates2 = [dates2[k] for k in buySignalIndex2]
signalCloses2 = [closes2[k] for k in buySignalIndex2]

zippedList1 = list(zip(signalDates1,signalCloses1,maxCloseDates1,signalMaxCloses1,returnsCloses1,maxHighDates1,signalMaxHighs1,
                     returnsHighs1))

zippedList2 = list(zip(signalDates2,signalCloses2,maxCloseDates2,signalMaxCloses2,returnsCloses2,maxHighDates2,signalMaxHighs2,
                     returnsHighs2))

colList = ['Signal Date','Signal Close','Max Close Date','Max Close','Close Return','Max High Date','Max High','High Return']

Results1 = pd.DataFrame(zippedList1,columns = colList)
Results2 = pd.DataFrame(zippedList2,columns = colList)

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

del df1['PX_OPEN']
del df1['PX_HIGH']
del df1['PX_LOW']

del df2['PX_OPEN']
del df2['PX_HIGH']
del df2['PX_LOW']

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('SPX Index Buy Signals')
plt.plot(df1[fields[3]], label=fields[3])
plt.plot(df1[fields[4]], label=fields[4])
plt.plot(df1[fields[5]], label=fields[5])
plt.plot(df1[fields[6]], label=fields[6])
for i in range(0,len(SPX_Buy)):
    plt.axvline(x = SPX_Buy[i], color = 'b')
plt.legend(loc=1)
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('SPW Index Buy Signals')
plt.plot(df2[fields[3]], label=fields[3])
plt.plot(df2[fields[4]], label=fields[4])
plt.plot(df2[fields[5]], label=fields[5])
plt.plot(df2[fields[6]], label=fields[6])
for i in range(0,len(SPW_Buy)):
    plt.axvline(x = SPW_Buy[i], color = 'r')
plt.legend(loc=1)
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('SPX and SPW Buy Signals')
plt.plot(df1['PX_LAST'], label='SPX Index')
plt.plot(df2['PX_LAST'], label='SPW Index')
for i in range(0,len(SPX_Buy)):
    plt.axvline(x = SPX_Buy[i], color = 'b')
for i in range(0,len(SPW_Buy)):
    plt.axvline(x = SPW_Buy[i], color = 'r')
plt.legend(loc=1)
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('SPX Index Sell Signals')
plt.plot(df1[fields[3]], label=fields[3])
plt.plot(df1[fields[4]], label=fields[4])
plt.plot(df1[fields[5]], label=fields[5])
plt.plot(df1[fields[6]], label=fields[6])
for i in range(0,len(SPX_Sell)):
    plt.axvline(x = SPX_Sell[i], color = 'b')
plt.legend(loc=1)
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('SPW Index Sell Signals')
plt.plot(df2[fields[3]], label=fields[3])
plt.plot(df2[fields[4]], label=fields[4])
plt.plot(df2[fields[5]], label=fields[5])
plt.plot(df2[fields[6]], label=fields[6])
for i in range(0,len(SPW_Sell)):
    plt.axvline(x = SPW_Sell[i], color = 'r')
plt.legend(loc=1)
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('SPX and SPW Sell Signals')
plt.plot(df1['PX_LAST'], label='SPX Index')
plt.plot(df2['PX_LAST'], label='SPW Index')
for i in range(0,len(SPX_Sell)):
    plt.axvline(x = SPX_Sell[i], color = 'b')
for i in range(0,len(SPW_Sell)):
    plt.axvline(x = SPW_Sell[i], color = 'r')
plt.legend(loc=1)
plt.show()

Results1.to_excel(r'C:\Users\MI2\Desktop\Excel\SPX Signals.xlsx', index = True)
Results2.to_excel(r'C:\Users\MI2\Desktop\Excel\SPW Signals.xlsx', index = True)