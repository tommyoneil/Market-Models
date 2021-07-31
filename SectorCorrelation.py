# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:14:28 2021

@author: MI2
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
import statistics as stats
import math

dataPeriod = 'MONTHLY'
startDate = dt(1997,1,1)
today = dt.today()

mgr = dm.BbgDataManager() 

vbs = ['S5INFT Index','S5FINL Index','S5ENRS Index','S5HLTH Index',
           'S5COND Index','S5CONS Index','S5TELS Index','S5INDU Index',
           'S5UTIL Index','S5MATR Index'] #'S5RLST Index']

#vbs = {'XLF US Equity','XLE US Equity','XLV US Equity','XLI US Equity',...
#     'XLY US Equity','XLP US Equity','XLB US Equity','XLK US Equity',...
#     'XLU US Equity'}; #'IYR US Equity'

sids = mgr[vbs]
df1 = sids.get_historical(['PX_Last'],startDate,today,dataPeriod)
df1.columns = df1.columns.droplevel(1)
#df1 = df1.fillna(method = 'ffill')
df1 = df1[vbs]

nRows = len(df1['S5INFT Index'][1:])-1
nColumns = len(vbs)

deltaFrame = pd.DataFrame(np.zeros(df1.shape), index=df1.index[:])
for i in range(1,len(deltaFrame.index)):
    for j in range(0,df1.shape[1]):
        deltaFrame.iloc[i-1,j] = math.log(df1.iloc[i-1,j]/df1.iloc[i,j])

correlWindowSize = range(5,46,5)
#correlWindowSize = range(3,13,3)
nRows = len(deltaFrame.index)
nColumns = len(correlWindowSize)
rollingSectorCorrelation = pd.DataFrame(np.zeros([nRows, nColumns]), index=deltaFrame.index)

for cWS in range(0,nColumns):#9 columns
    for i in range(0,(nRows - correlWindowSize[cWS])):
        #print(deltaFrame.iloc[i:i+correlWindowSize[cWS],:])
        RMatrix = np.corrcoef(deltaFrame.iloc[i:i+correlWindowSize[cWS],:], rowvar = False)
        TU = np.triu(RMatrix)
        TUVect = TU[:]
        TUVect = TUVect[TUVect != 1]
        TUVect = TUVect[TUVect != 0]
        Med = stats.median(TUVect)
        rollingSectorCorrelation.iloc[i+correlWindowSize[cWS],cWS] = Med
    plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.title(str(correlWindowSize[cWS]))
    plotSec = pd.DataFrame(data=rollingSectorCorrelation.iloc[(correlWindowSize[cWS]+1):,cWS]).reset_index(drop=True)
    plt.plot(plotSec)
  
export = pd.DataFrame(data=rollingSectorCorrelation, index=deltaFrame.index)
export.to_excel(r'C:\Users\MI2\Desktop\Excel\SectorIdxCorrelation.xlsx', index = True)
