# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:55:45 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import tia.bbg.datamgr as dm
from datetime import date as dt
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

startDate = dt(2015,1,1)
endDate = dt(2021,11,30)
dataPeriod = 'MONTHLY'

numbPCs = 2

# frame = 'SBOI Indicators';
# vbs = ['WGTROVER Index','SBOICOMP Index','SBOICAPS Index','SBOICMPP Index',...
#     'SBOIEXPA Index','SBOIHIRE Index','SBOIPPNM Index','SBOIEMPL Index','SBOICAPX Index',...
#     'SBOIPROF Index','SBOITOTL Index','SBOISALC Index','SBOILOAN Index'];
# offsets = [0,11,13,11,19,16,17,12,16,18,19,18,17];

frame = 'SBOI Edited for date range'
vbs = ['WGTROVER Index','SBOICOMP Index','SBOICAPS Index','SBOICMPP Index',
    'SBOIEXPA Index','SBOIHIRE Index','SBOIPPNM Index','SBOIEMPL Index',
    'SBOIPROF Index','SBOITOTL Index','SBOISALC Index']
offsets = [0,11,13,11,19,16,17,12,18,19,18]

# frame = 'SBOI indirects'
# vbs = ['WGTROVER Index','SBOICAPS Index','SBOIEXPA Index','SBOIPPNM Index',
#     'SBOIPROF Index','SBOITOTL Index','SBOISALC Index']
# offsets = [0,13,19,17,18,19,18]

# frame = 'SBOI Edited for date range AHE Target'
# vbs = ['USHEYOY Index','SBOICAPS Index','SBOISALC Index','SBOIPPNM Index',
#     'SBOIPROF Index','SBOIHIRE Index','SBOICMPP Index']
# offsets = [0,24,28,26,28,24,21]

# frame = 'SBOI Edited for date range'; #Quarterly
# vbs = ['COSYNFRM Index','SBOICOMP Index','SBOICAPS Index','SBOICMPP Index',
#     'SBOIEXPA Index','SBOIHIRE Index','SBOIPPNM Index','SBOIEMPL Index',
#     'SBOIPROF Index','SBOITOTL Index','SBOISALC Index']
# offsets = [0,4,4,4,6,5,6,4,6,6,6]

mgr = dm.BbgDataManager() 

sids = mgr[vbs]

df1 = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)
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
plt.plot(np.array(rotated.iloc[max(offsets)-1:,0]), color='g')
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
ex3 = pd.DataFrame(data=np.array(YPC), index=df1.index[7:])
ex4 = pd.DataFrame(data=np.array(YPC), index=df1.index[7:])

ex1.to_excel(r'C:\Users\MI2\Desktop\Excel\ATLevModel.xlsx', index = True)
ex2.to_excel(r'C:\Users\MI2\Desktop\Excel\ATLevPCAModel.xlsx', index = True)
ex3.to_excel(r'C:\Users\MI2\Desktop\Excel\ATLPCAModel.xlsx', index = True)
ex4.to_excel(r'C:\Users\MI2\Desktop\Excel\USHEPCModel.xlsx', index = True)