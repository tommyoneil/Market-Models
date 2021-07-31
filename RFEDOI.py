# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:48:09 2021

@author: MI2
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler #normalization

#Order-Inventory Gap
#Philly / Kansas / Richmond
regfed = ['OUTFNOF Index', 'OUTFIVF Index', 'KCLSVORD Index', 'KCLSIFIN Index', 'DFEDVNO Index',  'DFEDFING Index', 
          'EMPRNEWO Index','EMPRINVT Index', 'RCHSBANO Index', 'RCHSILFG Index']

startDate = dt(1990,1,1)
today = dt.today()
endDate = today+relativedelta(days=300)

mgr = dm.BbgDataManager() 

sids = mgr[regfed]

df1 = sids.get_historical(['PX_Last'],startDate,endDate)
df1.columns = df1.columns.droplevel(1)
#df1 = df1.dropna()
df1 = df1.fillna(method='ffill')
df1 = df1[regfed]

num = int(len(regfed)/2)
rd = pd.DataFrame(np.zeros([len(df1.index),num]), index=df1.index)
rd[0] = df1[regfed[0]][:] - df1[regfed[1]][:]
rd[1] = df1[regfed[2]][:] - df1[regfed[3]][:]
rd[2] = df1[regfed[4]][:] - df1[regfed[5]][:]
rd[3] = df1[regfed[6]][:] - df1[regfed[7]][:]
rd[4] = df1[regfed[8]][:] - df1[regfed[9]][:]
    
#rd = rd.dropna() #have to drop nans for PCA, but not for other operations

gapSum = rd.sum(axis=1,skipna=True)

#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(rd)
#principalDF = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'], index=rd.index)
#print(principalDF)

mean = gapSum.rolling(window=60).mean()
std = gapSum.rolling(window=60).std()
oiGap = ((gapSum - mean) / std)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(oiGap)
plt.show()

oiGap.to_excel(r'C:\Users\MI2\Desktop\Excel\RFEDOI.xlsx', index = True)

# 3 Feds order models
rfedOrdTickers = ['OUTFNOF Index', 'RCHSBANO Index', 'KCLSVORD Index']

sids2 = mgr[rfedOrdTickers]
endDate2 = today+relativedelta(days=30)
df2 = sids2.get_historical(['PX_Last'],dt(1990,1,1),endDate2) #put back in endDate2
df2.columns = df2.columns.droplevel(1)
df2 = df2.dropna()

nDate = pd.DataFrame(data=df2[:][:])

winSize = 60

zStorage3 = pd.DataFrame(np.zeros([len(df2.index), len(rfedOrdTickers)]))

for i in range(winSize,len(df2.index)):
    one = df2.iloc[i,:].values
    two = df2.iloc[i-winSize:i,:].mean(axis=0, skipna=True).values
    top = one - two
    bottom = df2.iloc[i-winSize:i,:].std(axis=0, skipna=True).values
    zStorage3.iloc[i,:] = top/bottom

fedZScore = zStorage3[winSize:]
fedZScore2 = pd.DataFrame()
fedZScore2['MEAN'] = fedZScore.mean(axis=1)

scaler = MinMaxScaler()
fedNorm = scaler.fit_transform(fedZScore2)

final = pd.DataFrame(data=fedNorm, index=zStorage3.index[winSize:])

zDate = zStorage3.index[winSize:]

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(final, color = 'r') 
plt.show()

final.to_excel(r'C:\Users\MI2\Desktop\Excel\PMIFed3.xlsx', index = True)