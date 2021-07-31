# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:51:53 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from business_calendar import Calendar
from YoY import *

dataPeriod = "QUARTERLY"
startDate = dt(1980,1,1)
endDate = dt(2023,1,1)
today = dt.today()

mgr = dm.BbgDataManager() 

ticker = ['TSLA US Equity'] #only works with a single security

sids = mgr[ticker]
sids.PX_LAST
df1 = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)

df1.columns = df1.columns.droplevel(1) #find way around this

df1 = df1.fillna(method = 'ffill')

#dates = pd.date_range(today.strftime("%Y%m%d"), periods=5) #gives the date range
cal = Calendar() #variable for calendar function
last_idx = df1.index[-1].date() #last date before future dates, today
idx_list = [] #creates a list to add date too
#c number of business days (excludes wknds and holidays) between start and end

if dataPeriod == 'DAILY':
    for i in range(0,days_in_between):
        weekdays = pd.DatetimeIndex([cal.addbusdays(last_idx, i)]) #creates dt index
        idx_list.append(weekdays[0]) #appending just the value so weekdays[0], adds future dates to column
    
if dataPeriod == 'WEEKLY': #get last day of each week
    fridays = pd.date_range(today, endDate, freq='W-FRI')
    for i in range(0,len(fridays)):
        idx_list.append(fridays[i])
        
if dataPeriod == 'MONTHLY':
    Mdiff = (endDate.year - today.year) * 12 + (endDate.month - today.month)
    months = pd.date_range(today, endDate, freq='M')
    for i in range(1, Mdiff):
        idx_list.append(months[i]) #idx_list.append(quarters) #appends last date of each quarter between current date and end date
   
if dataPeriod == 'QUARTERLY': #get last day of each quarter
    currentQuarter = pd.Timestamp(today).quarter
    endQuarter = pd.Timestamp(endDate).quarter
    Qdiff = (((endDate.year - today.year) * 4) + (endQuarter - currentQuarter)) #gets number of future quarters for range in loop below
    quarters = pd.date_range(today, endDate, freq='Q')
    for i in range(1,Qdiff):
        idx_list.append(quarters[i]) #idx_list.append(quarters) #appends last date of each quarter between current date and end date
            
idx_list.extend(df1.index) #extend the list with index
df2 = df1.reindex(idx_list).sort_index() #reindex the df and sort the index
df3 = df2.fillna(method = 'ffill')

output = YoY_Calc(dataPeriod, df3)

df3.to_excel(r'C:\Users\MI2\Desktop\Excel\BCOM Export.xlsx', index = True) #exports to excel

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df3['TSLA US Equity'],label='TSLA US Equity')
plt.plot(output,label='YoY')
plt.legend(loc=1)