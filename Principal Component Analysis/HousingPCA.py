# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:10:30 2021

@author: MI2
"""
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from sklearn.decomposition import PCA
from Norm import *

dataPeriod = "MONTHLY"
startDate = dt(1962,1,1)
endDate = dt(2021,1,3)

mgr = dm.BbgDataManager() 

vbs = ['SPCS20Y% Index','ETSLMSUP Index','NHSLSUPP Index','ETSLYOY Index',
       '.MBAYOY Index']
   
# vbs = ['C A Comdty','CCA Comdty','S A Comdty','SBA Comdty','LC1 Comdty',
#        'KCA Comdty','LHA Comdty','W A Comdty','FCA Comdty','CTA Comdty']

sids = mgr[vbs]

df1 = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)

df1.columns = df1.columns.droplevel(1)

df1 = df1.fillna(method = 'ffill')
df1 = df1.dropna()

df1 = normal_cal(df1)
print(df1)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df1)
principalDF = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'], index=df1.index)
print(principalDF)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(principalDF['Principal Component 1'],label='Principal Component 1')
plt.plot(principalDF['Principal Component 2'],label='Principal Component 2')
plt.legend(loc=1)

principalDF['Principal Component 1'].to_excel(r'C:\Users\MI2\Desktop\Excel\OwnerEqRentPCA.xlsx', index = True)
#principalDF['Principal Component 1'].to_excel(r'C:\Users\MI2\Desktop\Excel\DBAPCA.xlsx', index = True)