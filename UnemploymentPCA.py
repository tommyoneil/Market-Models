# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:22:24 2021

@author: MI2
"""
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from sklearn.decomposition import PCA
from dateutil.relativedelta import relativedelta
from Norm import *

dataPeriod = "MONTHLY"
startDate = dt(1962,1,1)
today = dt.today()
endDate = today+relativedelta(days=1000)

mgr = dm.BbgDataManager() 

vbs = ['USEMPTSW Index', 'CONCJOBH Index',
       'SBOIHIRE Index',  'ADP SMLL Index',
       'ADP SML Index', 'ADP MED Index', 'ADP LGE Index',
       'ADP ELGE Index', 'PRUSTOT Index']
#    , 'JOLTOPEN Index', 'JOLTLAYS Index',
#       'JOLTQUIS Index'};'ETI INDX Index']

sids = mgr[vbs]

df1 = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)

df1.columns = df1.columns.droplevel(1)
df1 = df1[vbs]
df1 = df1.dropna()

# for tgt in range(0,len(vbs)): 
#     #srTall = sids.get_historical(['PX_LAST'],shadowRateTargetNames[tgt],startDate,endDate)
#     sids = mgr[vbs[tgt]]
#     spxnacm = sids.get_historical(['PX_LAST'], startDate, endDate, dataPeriod);
#     print(vbs[tgt])
#     print(spxnacm[-1:1])


df1 = normal_cal(df1);

pca = PCA(n_components=1)
principalComponents = pca.fit_transform(df1)
principalDF = pd.DataFrame(data=principalComponents, columns=['PC 1'], index=df1.index)

final = pd.DataFrame(principalDF['PC 1'])

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(principalDF['PC 1'])
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df1)

final.to_excel(r'C:\Users\MI2\Desktop\Excel\UnemploymentPCA.xlsx', index = True)