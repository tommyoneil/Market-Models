# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:08:45 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import tia.bbg.datamgr as dm
from datetime import date as dt
from xbbg import blp

startDate = dt(2000,1,1)
today = dt.today()
endDate = dt.today()
dataPeriod = 'DAILY'
IdxMem = pd.read_excel(r'C:\Users\MI2\Desktop\Excel\CCMP IDs.xlsx')
MemList = list(IdxMem['ID()'])
mgr = dm.BbgDataManager() 
sids = mgr[MemList[0]]
Mkt_Cap = sids.get_historical(['CUR_MKT_CAP'],today, endDate)
Mkt_Cap = Mkt_Cap[IdxMem]
Mkt_Cap.columns = Mkt_Cap.columns.droplevel(1)

#for i in IdxMem['ID()'):
   #if Mkt_Cap[i][0] < 300:
        #Mkt_Cap.pop(i)