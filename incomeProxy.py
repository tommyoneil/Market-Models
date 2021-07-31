# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:39:03 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from business_calendar import Calendar
from YoY import *

dataPeriod = 'MONTHLY'
startDate = dt(1995,1,1)
endDate = dt.today()

#inputs = {'AHE TOTL Index','USAWTOT Index'};
inputs = ['USHETOT Index','USAWTOT Index']

mgr = dm.BbgDataManager() 

sids = mgr[inputs]

wage = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)

wage.columns = wage.columns.droplevel(1) #find way around this

prod = wage.iloc[:,0].multiply(wage.iloc[:,1])
prodYoY = YoY_Calc(dataPeriod,prod)

final = pd.DataFrame(data=prodYoY.iloc[12:], index=wage.index[12:])
final.to_excel(r'C:\Users\MI2\Desktop\Excel\IncomeProxy.xlsx', index = True)