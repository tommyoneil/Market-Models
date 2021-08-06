# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:48:18 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from dateutil.relativedelta import relativedelta
import math
import statsmodels.api as sm
import statistics
from Norm import *
from fitnetFunction import *
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import MinMaxScaler

startDate = dt(1990,1,1)
today = dt.today()
endDate = today + relativedelta(days=1000)
today = dt.today()

unempMdlNames = ['USURTOT Index', 'USEMPTSW Index', 'CONCJOBH Index',
                'SBOIHIRE Index', 'ETI INDX Index', 'ADP SMLL Index',
                'ADP SML Index', 'ADP MED Index', 'ADP LGE Index',
                'ADP ELGE Index', 'PRUSTOT Index', 'JOLTOPEN Index', 'JOLTLAYS Index',
                'JOLTQUIS Index']
                       
#             unempMdlNames = ['USURTOT Index', 'USEMPTSW Index', 'CONCJOBH Index',
#                 'SBOIHIRE Index', 'ETI INDX Index', 'ADP SMLL Index',
#                 'ADP SML Index', 'ADP MED Index', 'ADP LGE Index',
#                 'ADP ELGE Index',  'JOLTOPEN Index', 'JOLTLAYS Index',
#                 'JOLTQUIS Index']

#unempMdlNames = ['USURTOT Index', 'USEMPTSW Index','SBOIHIRE Index', 'ETI INDX Index']:
    
mgr = dm.BbgDataManager() 

sids = mgr[unempMdlNames]

df1 = sids.get_historical(['PX_Last'],startDate, endDate)
df1.columns = df1.columns.droplevel(1)
df1 = df1.fillna(method='ffill')
df1 = df1.dropna() #Last three indices have nans for 6-30-21, so dropped
df1 = df1[unempMdlNames]

print('The Latest date is ' + str(df1.index[-1]))
print('The Earliest date is ' + str(df1.index[0]))

# Input checking  df1.iloc[0:]
df2 = df1.iloc[:,0:-2]
df2['UnempTrans'] = np.divide(df1.iloc[:,-1], df1.iloc[:,-2])
df2 = df2.reset_index(drop=True)

[inputM, inputN] = df2.shape

chartRow = math.ceil(inputN/3)

for i in range(0, inputN):
    plt.subplot(chartRow,3,i+1)
    plt.plot(df2.iloc[:,i])
    plt.title(unempMdlNames[i])
    plt.tight_layout(pad=.25)
plt.show()

#df1 = df1.reset_index(drop=True)

unempCut = df1.iloc[0:140, :]

regr = sm.OLS(normal_cal(pd.DataFrame(data=unempCut.iloc[4:,0]).reset_index(drop=True)),normal_cal(pd.DataFrame(data=unempCut.iloc[0:-4,1:]).reset_index(drop=True))).fit()
regWeight = regr.params

regr2 = sm.OLS(np.array(unempCut.iloc[4:,0]),np.array(unempCut.iloc[0:-4,1:])).fit()
rawRegWeight = regr2.params

shift = 4 #shifts graph over by inserting NaN 

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(normal_cal(pd.DataFrame(data=df1.iloc[:,0]).reset_index(drop=True)))
regFrame = np.dot(normal_cal(pd.DataFrame(data=df1.iloc[:,1:]).reset_index(drop=True)),regWeight)
regFrame = pd.DataFrame(data=regFrame)
final = pd.DataFrame([[np.nan * len(regFrame.columns)] for i in range(0,shift)], columns=regFrame.columns)
final = final.append(regFrame, ignore_index=True)
plt.plot(final, color='r')
plt.title('Regression')
plt.show()

# Charting the subs
df3 = df1.reset_index(drop=True)
for i in range(0,len(unempMdlNames)-1):
    plt.subplot(4,4, i+1)
    plt.plot(df3.iloc[:,i+1])
    plt.tight_layout(pad=.25)
plt.show()

# Model testing
for i in range(1,len(df1.iloc[0,:])):
    plt.subplot(4,4,i)
    plt.plot(df3.iloc[:,i])
    plt.title(unempMdlNames[i])
    plt.tight_layout(pad=.25)
plt.show()

tip = pd.DataFrame(data=np.array(df1.iloc[1:,1:]), index=range(0,len(df1.iloc[1:,1:])))
bip = pd.DataFrame(data=np.array(df1.iloc[0:-1, 1:]), index=range(0,len(df1.iloc[0:-1, 1:])))
Sub = tip.subtract(bip)
unempMoM = np.divide(np.array(Sub), np.array(df1.iloc[0:-1, 1:]))

for j in range(0,len(unempMoM[0,:])):
    plt.subplot(4,4,j+1)
    plt.plot(unempMoM[:,j])
    plt.title(unempMdlNames[j])
    plt.tight_layout(pad=.25)
plt.show()

# The MoM of ADP resembles the NFP 

## ANN Traning
tTimes = 100
cutoff = len(df1.iloc[:, 0]) - 8

#df3 = df1.iloc[:,:-2]
#df3['UnempTrans'] = np.divide(df1.iloc[:,-1],df1.iloc[:,-2])
#df3 = df3.reset_index(drop=True)
#df1 = unempTrans['UnempTrans']
mdlData = df1.iloc[0:cutoff, :]

inputs = mdlData.iloc[0:-4, 1:]
targets = mdlData.iloc[4:, 0]

mdlStorage = np.empty([tTimes, len(df1.iloc[:,0])])

finalUnemp = pd.DataFrame(data=df1['USURTOT Index']).reset_index(drop=True)

#nanDf = pd.DataFrame(data=df1).reset_index(drop=True)
#nanDf.index = nanDf.index + shift
#modDf = nanDf.reindex(np.arange(len(df1.index)+shift)).dropna()

num_cores = multiprocessing.cpu_count()
results = (Parallel(n_jobs=num_cores, verbose=50)(delayed(fitnetFunction)(inputs, targets, df1.iloc[:,1:],5)for i in range(tTimes)))
#does it apply shift to model to predict, both USURTOT and the predicted values are 197 in length
resultAr = np.array(results) #or does it offset the first four and predict last four so length should remain the same
mdlResult = np.mean(resultAr, axis=0)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(np.array(finalUnemp['USURTOT Index'][4:]), color='blue')
plt.plot(mdlResult, color='r')
plt.title('Cut off ' + str(cutoff))
plt.show()

# save to excel 
# print(str(df1.index[-1]))
data = {'Date': df1.index, 'MDL Result': mdlResult}
export1 = pd.DataFrame(data)
export1.to_excel(r'C:\Users\MI2\Desktop\Excel\UnempnnIndex.xlsx', index = True)

# In depth research on the subs

# Percentage change chart

[pctM, pctN] = df1.shape
tip = pd.DataFrame(data=np.array(df1.iloc[1:,:]), index=range(0,len(df1.iloc[1:,:])))
bip = pd.DataFrame(data=np.array(df1.iloc[0:-1, :]), index=range(0,len(df1.iloc[0:-1, :])))
Sub = tip.subtract(bip)
subPctChange = np.divide(np.array(Sub), np.array(df1.iloc[0:-1, :]))

nCharts = 4

nSlides = math.ceil(pctN/nCharts)
for i in range(0, nSlides): 
    for j in range(0,nCharts):
        if 4*(i-1)+j <= pctN:
            subF = pd.DataFrame(data=subPctChange[0:,4*(i-1)+j]).reset_index(drop=True)
            colP = pd.DataFrame(data=df1.iloc[1:,4*(i-1)+j]).reset_index(drop=True)
            fig, ax = plt.subplots()
            ax.plot(subF, color='r')
            ax.set_ylabel('SubPctChange', color='r')
            ax2=ax.twinx()
            ax2.plot(colP, color='blue')
            ax2.set_ylabel('UnempAll', color='blue')
            plt.title(unempMdlNames[4*(i-1)+j])
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(subPctChange)
plt.legend(unempMdlNames)
plt.title('MoM PctChange')
plt.show()

# Simple Regression on the first 3 subs

unemp3 = df1.iloc[:,0:9]

unemp3WeightOLS = sm.OLS(normal_cal(pd.DataFrame(data=unemp3.iloc[4:,0]).reset_index(drop=True)), normal_cal(pd.DataFrame(data=unemp3.iloc[0:-4,1:9]).reset_index(drop=True))).fit()
unemp3Weight = unemp3WeightOLS.params
unemp3Weight2OLS = sm.OLS(normal_cal(pd.DataFrame(data=unemp3.iloc[:, 0]).reset_index(drop=True)),normal_cal(pd.DataFrame(data=unemp3.iloc[:, 1:9]).reset_index(drop=True))).fit()
unemp3Weight2 = unemp3Weight2OLS.params

#unemp3Weight

unempFrame = np.dot(normal_cal(pd.DataFrame(data=unemp3.iloc[:,1:9]).reset_index(drop=True)),unemp3Weight)
unempFrame = pd.DataFrame(data=unempFrame)
nanDf2 = pd.DataFrame(data=unempFrame).reset_index(drop=True)
nanDf2.index = nanDf2.index + shift
final = nanDf2.reindex(np.arange(len(unempFrame.index)+shift)).dropna()
unempFrame2 = np.dot(normal_cal(pd.DataFrame(data=unemp3.iloc[:,1:9]).reset_index(drop=True)),unemp3Weight2)
unempFrame2 = pd.DataFrame(data=unempFrame2)
nanDf3 = pd.DataFrame(data=unempFrame2).reset_index(drop=True)
nanDf3.index = nanDf3.index + shift
final2 = nanDf3.reindex(np.arange(len(unempFrame2.index)+shift)).dropna()
plt.figure(figsize=[15,10])
plt.grid(True)  
plt.plot(normal_cal(pd.DataFrame(data=unemp3.iloc[:,0]).reset_index(drop=True)), color='k')
plt.plot(final, color = 'r') 
plt.plot(np.array(final2), color = 'g')
plt.show()

corr1 = df1.iloc[4:,0].corr(unempFrame2.iloc[0:-4]) 
corr2 = df1.iloc[4:,0].corr(unempFrame.iloc[4:])

## Final section
for i in range(0,10):
  plotF = pd.DataFrame(data=df1.iloc[:,i]).reset_index(drop=True)
  plt.subplot(3,4,i+1)
  plt.plot(plotF)
  plt.tight_layout(pad=.25)
plt.show()
'''
#EXTRAS
# Initial test
# overall index comp
s = pd.Series(data=[1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
datc = pd.DataFrame(0, index=s.index, columns=s.index, dtype=s.dtype)
[ttlM, ttlN] = df1.shape
np.fill_diagonal(datc.values, pd.Series(data=[1, 1, 1, -1, -1, -1, -1, -1, -1, -1]))
unempAll2 = np.dot(df1,datc)

triComp = np.nan(ttlM-1, ttlN):
for i in range(0,ttlN):
    for j in range(1,ttlM):
        if unempAll2(j,i) > unempAll2(j-1,i)
            triComp.iloc[j-1,i] = 1
        elif (unempAll2.iloc[j,i] < unempAll2.iloc[j-1,i]:
            triComp.iloc[j-1,i] = -1
        else:
            triComp.iloc[j-1,i] = 0

triCorr = triCom.iloc[:,0].corr(triComp.iloc[:,1:])
triCorrLagged = triComp.iloc[4:,0].corr(triComp.iloc[0:-4,1:])
    
dirCorr = unempAll2.ilocp[:,0].corr(unempAll2.iloc[:,1:])
dirCorrLagged = unempAll2.iloc[4:,0].corr(unempAll2.iloc[0:-4,1:])

plt.plot(dirCorr)
plt.plot(dirCorrLagged, color = 'r')
plt.title('dirCorr')
plt.show()

## Section for ordinary regression
# Regression with focus on ADP Data

regLead = 4:
regCut = 140:

adpData = df1.iloc[:,5:9] #may have to change ilocs to locs or around
regGoal = df1.iloc[:,0]

adpWeight = sm.OLS(np.array(adpData.iloc[0:-regLead, :]), np.array(regGoal.iloc[0+regLead:]))

# regResult = adpData() 
# 
# plt.plot(np.dot(adpData,adpWeight))
# plt.title('Regression Results')

regRawResult = np.dot(df1.iloc[:,1:],rawRegWeight)
test123 = statistics.mean(mdlStorage)

unempEnsem = df1.iloc[:,0]

ensemBunch = regRawResult.concat(test123)
ensemWeight = sm.OLS(np.array(ensemBunch.iloc[0:125]),np.array(unempEnsem.iloc[4:129]))

plt.plot(unempEnsem)
shift = 4
finDat = np.dot(ensemBunch, ensemWeight)
final = pd.DataFrame([[np.nan * len(finDat.columns)] for i in range(0,shift)], columns=finDat.columns)
plt.plot(final, color = 'r')
plt.show()
'''