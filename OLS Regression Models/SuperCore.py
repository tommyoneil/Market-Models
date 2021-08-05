# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:22:23 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import tia.bbg.datamgr as dm
from datetime import date as dt
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

startDate = dt(1980,1,1)
endDate = dt(2018,8,31)

numbPCs = 2

# All Super Core Indicators threshold of 0.8
# frame = 'All Super Core'
# vbs = ['BCMPEASC Index',
#     #one month lag 2-6
#     'EPB0B8EA Index','EPB236EA Index','EPB27IEA Index',
#     'EPB28EEA Index','EPY291EA Index',
#     #short term lags (3-6 mo) 7-12
#     'EPY028EA Index','EPY281EA Index','EPB028EA Index','EPB023EA Index',
#     'PPCGEMUY Index','EPB27AEA Index',
#     #medium term lags (8-12 mo) 13-18
#     'EPB24REA Index','EPY025EA Index','EPB251EA Index','EPY25MEA Index',
#     'EPY25CEA Index','EPY16HEA Index',
#     #long term lags (14-17 mo) 19-22
#     'EPY32GEA Index','EPY027EA Index','EPY323EA Index','EPY13HEA Index']
# offsets = [0,1,1,1,1,1,3,3,4,6,6,6,8,8,8,8,10,12,14,15,16,17]
# 
# frame = 'Short term out Super Core'
# vbs = ['BCMPEASC Index',
#      #short term lags (3-6 mo)  
#      'EPY028EA Index','EPY281EA Index','EPB028EA Index','EPB023EA Index',
#      'PPCGEMUY Index','EPB27AEA Index',
#      #medium term lags (8-12 mo)
#      'EPB24REA Index','EPY025EA Index','EPB251EA Index','EPY25MEA Index',
#      'EPY25CEA Index','EPY16HEA Index',
#      #long term lags (14-17 mo) 'EPY13HEA Index'
#      'EPY32GEA Index','EPY027EA Index','EPY323EA Index','USD Curncy']
# offsets = [0,3,3,4,6,6,6,8,8,8,8,10,12,14,15,16,17]
# 
# frame = 'Mid term out Super Core'
# vbs = ['BCMPEASC Index',
#      #medium term lags (8-12 mo)
#      'EPB24REA Index','EPY025EA Index','EPB251EA Index','EPY25MEA Index',
#      'EPY25CEA Index','EPY16HEA Index',
#      #long term lags (14-17 mo)
#      'EPY32GEA Index','EPY027EA Index','EPY323EA Index','EPY13HEA Index']
# offsets = [0,8,8,8,8,10,12,14,15,16,17]

# frame = 'long term Super Core'
# vbs = {'BCMPEASC Index',
#     #long term lags (14-17 mo)
#     'EPY32GEA Index','EPY027EA Index','EPY323EA Index','EPY13HEA Index'};
# offsets = [0,14,15,16,17]
# 
# frame = 'Mid term only Super Core'
# vbs = ['BCMPEASC Index',
#      #medium term lags (8-12 mo)
#      'EPB24REA Index','EPY025EA Index','EPB251EA Index','EPY25MEA Index',
#      'EPY25CEA Index','EPY16HEA Index']
# offsets = [0,8,8,8,8,10,12]

# frame = 'Short term only Super Core'
# vbs = ['BCMPEASC Index',
#      #short term lags (3-6 mo)  
#      'EPY028EA Index','EPY281EA Index','EPB028EA Index','EPB023EA Index',
#      'PPCGEMUY Index','EPB27AEA Index']
# offsets = [0,3,3,4,6,6,6,]

# frame = 'Short and mid Super Core'
# vbs = ['BCMPEASC Index',
#      #short term lags (3-6 mo)  
#      'EPY028EA Index','EPY281EA Index','EPB028EA Index','EPB023EA Index',
#      'PPCGEMUY Index','EPB27AEA Index',
#      #medium term lags (8-12 mo)
#      'EPB24REA Index','EPY025EA Index','EPB251EA Index','EPY25MEA Index',
#      'EPY25CEA Index','EPY16HEA Index']
# offsets = [0,3,3,4,6,6,6,8,8,8,8,10,12]

frame = 'Short mid Super Core'
vbs = ['BCMPEASC Index',
       #short term lags (3-6 mo)  
       'EPB023EA Index','PPCGEMUY Index','EPB27AEA Index',
       #medium term lags (8-12 mo)
       'EPB24REA Index','EPY025EA Index','EPB251EA Index','EPY25MEA Index',
       'EPY25CEA Index','EPY16HEA Index']
offsets = [0,6,6,6,8,8,8,8,10,12]

mgr = dm.BbgDataManager() 

sids = mgr[vbs]

df1 = sids.get_historical(['PX_Last'],startDate,endDate)
df1.columns = df1.columns.droplevel(1)
df1 = df1[vbs]
df1 = df1.dropna()

rotated = pd.DataFrame(np.ones([len(df1.index),len(vbs)]))

for i in range(0,len(vbs)):
   rotated.iloc[:,i] = np.roll(df1.iloc[:,i],offsets[i])

betafull = sm.OLS(rotated.iloc[max(offsets)-1:,0],rotated.iloc[max(offsets)-1:,1:]).fit()

print(betafull.summary())

y1 = rotated.iloc[max(offsets)-1:,1:]
y2 = rotated.iloc[0:min(offsets[1:])-1,1:]
pieces = [y1,y2]
Yfull = np.dot(pd.concat([y1,y2], ignore_index=True), betafull.params)

Y = rotated.iloc[max(offsets)-1:,0]
Xt = Yfull[0:len(Y)]
R3 = np.corrcoef(Y,Xt);
plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(rotated.iloc[max(offsets)-1:,0], color='g')
plt.plot(Yfull)
plt.title(frame + ' corr:' + str(R3[0,1]))
plt.show()

PC1 = rotated.iloc[max(offsets)-1:,1:]
PC2 = rotated.iloc[0:min(offsets[1:]),1:]
PCAPrep = pd.concat([PC1,PC2], ignore_index=True)
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(PCAPrep)
principalDF = pd.DataFrame(data=principalComponents, columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'])

b1 = principalDF.iloc[0:-min(offsets[1:]),0:numbPCs]
nRows = len(principalDF.iloc[0:-min(offsets[1:])].index)
b1['0'] = np.ones([nRows,1])
Y = Y.reset_index(drop=True)
beta = sm.OLS(Y,b1, ignore_index=True).fit()

c1 = principalDF.iloc[:,0:numbPCs]
nRows2 = len(principalDF.iloc[:,0:numbPCs].index)
c1['0'] = np.ones([nRows2,1])
YPC1 = np.dot(c1, beta.params)
YPC = pd.DataFrame(YPC1)

R = np.corrcoef(Y,YPC.iloc[0:-min(offsets[1:]),0])

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(Y, color='k')
plt.title(frame +' PCs ' + str(numbPCs) + ' corr:' + str(R[0,1]))
plt.plot(YPC, color='r')
plt.show()

YLev = Yfull - np.roll(Yfull,14)
YLev = YLev[14:]
YPLev = np.array(YPC) - np.roll(YPC,14)
YPLev = YPLev[14:]

dateList = df1.index.date

ex1 = pd.DataFrame(data=YLev, index=dateList[-len(YLev)-1:-1])
ex2 = pd.DataFrame(data=YPLev, index=dateList[-len(YPLev):])
ex3 = pd.DataFrame(data=np.array(YPC), index=df1.index[5:])

ex1.to_excel(r'C:\Users\MI2\Desktop\Excel\ATLevModel.xlsx', index = True)
ex2.to_excel(r'C:\Users\MI2\Desktop\Excel\SuperCorePCAModel.xlsx', index = True)
ex3.to_excel(r'C:\Users\MI2\Desktop\Excel\SuperCorePCAModel2.xlsx', index = True)