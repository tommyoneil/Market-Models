# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:31:26 2021

@author: MI2
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from sklearn.decomposition import PCA
from Norm import *
from columnDrop import *
from YoY_AddOn import *
import statsmodels.api as sm
import seaborn as sns; sns.set_theme()
from PCA_Coeff import *

dataPeriod = 'MONTHLY'
startDate = dt(2001,1,1)
endDate = dt(2017,12,14)

EUIGInputs= pd.read_excel(r'C:\Users\MI2\Desktop\MATLAB\CIXIsfromMatlab\EuroCoinInputs.xlsx') #put in user path for xlsx
EUIGNames = EUIGInputs.iloc[:,0]
Categories =  EUIGInputs.iloc[:,3]
CategoriesSet = Categories.unique()
CategoriesSet = sorted(CategoriesSet)

infTgts = ['BCMPEASC Index','CPEXEMUY Index','ECCPEMUY Index',
    'GDBR10 Index','EUIPEMUY Index']

mgr = dm.BbgDataManager() 
sids = mgr[EUIGNames]

EUIGAll = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)
EUIGAll.columns = EUIGAll.columns.droplevel(1)
EUIGAll = EUIGAll[EUIGNames]

EUIGAll1 = EUIGAll.iloc[:,:95].dropna().reset_index(drop=True)
EUIGAll2 = EUIGAll.iloc[62:226,95:]
EUIGAll3 = columnDrop(EUIGAll2)
    
EUIGAll = pd.concat([EUIGAll1, EUIGAll3], axis=1)

transformSwitches = EUIGInputs.iloc[:,2]
EUIGAllTrans = YoY_AddOn(dataPeriod,EUIGAll,transformSwitches) 

EUIGAllnc = normal_cal(EUIGAllTrans.iloc[12:,:].reset_index(drop=True))

# EUIGAllnc = EUIGAllTrans(13:end,:);
[M,N] = EUIGAllnc.shape

for i in range(1,M):
    for j in range(0,N):
        if pd.isna(EUIGAllnc.iloc[i,j]) == True:
            print([i,j])
            EUIGAllnc.iloc[i,j] = EUIGAllnc.iloc[i-1,j]

sids = mgr[infTgts]
HLAll = sids.get_historical(['PX_Last'], startDate, endDate, dataPeriod)
HLAll.columns = HLAll.columns.droplevel(1)
HLAll = HLAll[infTgts]
HLAll2 = pd.DataFrame()

HLAll2 = columnDrop(HLAll)
    
HLAllnc = HLAll2.iloc[12:,:]
    #HLAllnc = normal_cal(HLAll)

## =======================================================================
# PCA and Analysis
pca = PCA(n_components=len(EUIGAllnc)-1)
principalComponents1 = pca.fit_transform(EUIGAllnc)

EUIScore = pd.DataFrame(data=principalComponents1).reset_index(drop=True)

Coeffs = PCA_Coeff(EUIGAllnc)

EUICoeffs = pd.DataFrame(Coeffs)

numbPCs = 1
for PCnumb in range(0,4):
    lagsize = 0
    for tgtIndex in range(0,len(infTgts)):
        takenPCs = pd.DataFrame(EUIScore.iloc[0:,PCnumb:PCnumb+(numbPCs)]).reset_index(drop=True)
        takenPCs['1'] = np.ones([len(EUIScore.iloc[0:,0]),1])
        tgt = HLAllnc.iloc[lagsize:len(EUIScore.iloc[:,0])+lagsize,tgtIndex].reset_index(drop=True)
        beta = sm.OLS(tgt,np.array(takenPCs)).fit().params
        Y3 = np.dot(takenPCs,beta)
        plt.figure(figsize=[15,10])
        plt.grid(True)
        plt.plot(tgt);
        plt.title(infTgts[tgtIndex] + ' PC number: ' + str(PCnumb+1))
        plt.plot(Y3)
        plt.show()

for categoryID in range(0,len(CategoriesSet)):
    ctgidx = []
    for i in range(0,len(Categories)):
        if str(list(Categories)[i]) == str(CategoriesSet[categoryID]):
            ctgidx.append(1)
        else:
            ctgidx.append(0)
    ax = plt.axes()
    heater = sns.heatmap(EUICoeffs.iloc[ctgidx,0:2], ax=ax)
    ax.set_title(CategoriesSet[categoryID])
    plt.show()

ax = plt.axes()
overAllHeat = sns.heatmap(EUICoeffs.iloc[:,0:2], ax=ax)
ax.set_title(' Overall heat map')
plt.show()