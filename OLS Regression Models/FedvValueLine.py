# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:49:47 2021

@author: MI2
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
import statsmodels.api as sm
from YoY import *

dataPeriod = 'WEEKLY'
startDate = dt(2008,1,1)
endDate = dt(2018,1,31)

vbs = ['FARBFSRF Index','VALUG Index']

mgr = dm.BbgDataManager() 
sids = mgr[vbs]

df1 = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)
df1.columns = df1.columns.droplevel(1)
df1 = df1[vbs]

newDF = pd.DataFrame()

newDF[vbs[0]] = df1[vbs[0]][:].dropna().reset_index(drop=True)
newDF[vbs[1]] = df1[vbs[1]][:].dropna().reset_index(drop=True)

betaAndCorr = pd.DataFrame(data=np.ones([len(newDF.iloc[:,0]) - 20,2]))

for i in range(0,len(newDF.iloc[:,0]) - 20):
    X = pd.DataFrame(data=newDF.iloc[i:i+20,0])
    X['0'] = np.ones([20,1])
    Y = np.divide(newDF.iloc[i:i+20,1],1000)
    betaB = sm.OLS(np.array(Y),X.reset_index(drop=True), ignore_index=True).fit().params
    betaDF = pd.DataFrame(data=betaB)
    betaAndCorr.iloc[i,0] = betaDF.iloc[0,0]*1000000
    corrB = pd.DataFrame(data=np.corrcoef(X.iloc[:,0],Y))
    betaAndCorr.iloc[i,1] = corrB.iloc[1,0]

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(betaAndCorr.iloc[:,0])
plt.show()
plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(betaAndCorr.iloc[:,1])
plt.show()

newIndex = df1.index.to_period('W').drop_duplicates()

betaAndCorr = betaAndCorr.set_index(newIndex[20:])
final1 = betaAndCorr.iloc[:,0]
final2 = betaAndCorr.iloc[:,1]

final1.to_excel(r'C:\Users\MI2\Desktop\Excel\FedvValueGBeta.xlsx', index = True)
final2.to_excel(r'C:\Users\MI2\Desktop\Excel\FedvValueGCorr.xlsx', index = True)