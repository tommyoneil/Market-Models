# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:18:21 2021

@author: MI2
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
import statsmodels.api as sm

#dataPeriod = "DAILY"
startDate = dt(1988,1,1)
today = dt.today()

mgr = dm.BbgDataManager() 

vbs = ['USGG10YR Index','NAPMNEWO Index','CPI XYOY Index','USGG2YR Index','USGG3M Index']
        #factor to predict    #-------factors which have an impact on USGG10YR----independent variables
        #dependent variables
sids = mgr[vbs]
df1 = sids.get_historical(['PX_Last'],startDate,today) #dataPeriod)
df1.columns = df1.columns.droplevel(1)
df1 = df1.dropna()

Y = df1['USGG10YR Index']
X = df1.iloc[:,1:]
X2 = X
model = sm.OLS(Y,X).fit()
print(model.summary())
beta = model.params
Yhat = np.dot(X,beta) #multiply the independent variables by beta to find the model approximated value
YhatPlot = pd.DataFrame(Yhat, index=df1.index)
const = np.squeeze(np.ones([len(Y),1]).tolist())

X2['Xalt'] = const
Xalt = X2

model2 = sm.OLS(Y,Xalt).fit()
betaalt = model2.params
Yalt = np.dot(Xalt,betaalt)
YaltPlot = pd.DataFrame(Yalt, index=df1.index)
daysout = [51,101,151,201,len(Y.to_numpy().tolist())]

alternativeStorage = pd.DataFrame(np.squeeze(np.ones([len(Y.to_numpy().tolist()), len(daysout)])), index=df1.index)

for i in range(0,len(daysout)): 
    Yabbrev = Y.iloc[daysout[i]-50:daysout[i]] #iterates from 1-51, 51-101, 101-151, 151-201... for Y
    Xabbrev = X.iloc[daysout[i]-50:daysout[i],:] #same as abovebut for X and for all columns
    model3 = sm.OLS(Yabbrev,Xabbrev).fit() #takes ordinary least squares between Yabbrev and Xabbrev
    betaabbrev = model3.params #gets beta
    
    Yguestimate = np.dot(X,betaabbrev)         
    alternativeStorage.iloc[:,i] = Yguestimate
    YguestimatePlot = pd.DataFrame(Yguestimate, index=df1.index)
    plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.plot(YguestimatePlot) 
    plt.plot(Y,marker='o')

spread = alternativeStorage.max(axis=1) - alternativeStorage.min(axis=1)
alternativeStoragePlot = pd.DataFrame()
alternativeStoragePlot['Mean'] = alternativeStorage.mean(axis=1)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(Y,color='k')
plt.plot(YhatPlot,color='r') #has four columns which it will print and chart
plt.plot(YaltPlot,color='g') #has five columns which it will print and chart
plt.title('Y=Black, Yhat=Red, Yalt=Green')
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(alternativeStoragePlot)
plt.plot(spread,color='k')
plt.plot(Y,marker='o')
plt.show()