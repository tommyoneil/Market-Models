# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:17:08 2021

@author: MI2
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from sklearn.decomposition import PCA
import statsmodels.api as sm

dataPeriod = 'MONTHLY'
startDate = dt(2000,1,1)
endDate = dt(2018,12,14)

mgr = dm.BbgDataManager() 

EPFTickers = [#Primary Target, 1
    #'ECCPEMUY Index',...
    #'CPEXEMUY Index',...
    'ECCPEMUY Index',
    #3 month leaders, 2-23
    'EPFC11DE Index','EPF010DE Index','EPFC10DE Index','EPF0NDNL Index',
    'EPF03GNL Index','EPB109EA Index','EPB10XEA Index','EPY10XEA Index',
    'EPF104NL Index','EPF032BE Index','EPB10FEA Index','EPB105EA Index',
    'EPB10JEA Index','EPB027EA Index','EPY11MEA Index','EPB172EA Index',
    'EPY27FEA Index','EPB272EA Index','EPF106NL Index','EPD272EA Index',
    'EPH106EA Index','EPF10GNL Index',
    #6 month leaders, 24-28
    'EPH10FEA Index','EPF027ES Index','EPF10BEA Index','EPY15EEA Index',
    'EPF0KGBE Index',
    #12 month leaders, 29-33
    'EPF016AT Index','EPD25REA Index','EPF281IT Index','EPH25PEA Index',
    'EPF274IT Index',
    #18 month leaders, 34-36
    'EPF033AT Index','EPD32FEA Index','EPD321EA Index'
    ]

sids = mgr[EPFTickers]
EPFAll = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)
EPFAll.columns = EPFAll.columns.droplevel(1)
#EPFAll = EPFAll.fillna(method = 'ffill')
EPFAll = EPFAll.dropna()
EPFAll = EPFAll[EPFTickers]

#Inputs (leaders found using 0.6 correlation threshold


#Pull data and PCA groups
numbPCs = 2

PCADF = EPFAll.copy()

pca = PCA(n_components=22)
ThreeMScore = pd.DataFrame(data=pca.fit_transform(PCADF.iloc[:,1:23]))

pca = PCA(n_components=5)
SixMScore = pd.DataFrame(data=pca.fit_transform(PCADF.iloc[:,23:28]))

pca = PCA(n_components=5)
TwelveMScore = pd.DataFrame(data=pca.fit_transform(PCADF.iloc[:,28:33]))

pca = PCA(n_components=3)
EighteenMScore = pd.DataFrame(data=pca.fit_transform(PCADF.iloc[:,33:36]))



betaIn = pd.DataFrame(data=ThreeMScore.iloc[0:-3,0:numbPCs]).reset_index(drop=True)
betaIn['0'] = np.ones([len(EPFAll.iloc[3:,0]),1]
                      )
beta3 = sm.OLS(np.array(EPFAll.iloc[3:,0]),betaIn, ignore_index=True).fit().params
Y3In = pd.DataFrame(data=ThreeMScore.iloc[:,0:numbPCs])
Y3In['0'] = np.ones([len(ThreeMScore.iloc[:,0]),1])
Y3 = np.dot(Y3In, beta3)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(EPFAll.iloc[3:,0].reset_index(drop=True))
R3 = np.corrcoef(EPFAll.iloc[3:,0],Y3[0:-3])
plt.title('Only 3 month' + ' PCs ' + str(numbPCs) + ' corr: ' + str(R3[0,1]))
plt.plot(Y3)
plt.show()

betaIn2 = pd.DataFrame(data=SixMScore.iloc[0:-6,0:numbPCs]).reset_index(drop=True)
betaIn2['0'] = np.ones([len(EPFAll.iloc[6:,0]),1])
                      
beta6 = sm.OLS(np.array(EPFAll.iloc[6:,0]),betaIn2, ignore_index=True).fit().params
Y6In = pd.DataFrame(data=SixMScore.iloc[:,0:numbPCs])
Y6In['0'] = np.ones([len(SixMScore.iloc[:,0]),1],beta6)
Y6 = np.dot(Y6In, beta6)

R6 = np.corrcoef(EPFAll.iloc[6:,0],Y6[0:-6])
plt.figure(figsize=[15,10])
plt.grid(True)
plt.title('Only 6 month' + ' PCs ' + str(numbPCs) + ' corr: ' + str(R6[0,1]))
plt.plot(EPFAll.iloc[6:,0].reset_index(drop=True))
plt.plot(Y6)
plt.show()

betaIn3 = pd.DataFrame(data=TwelveMScore.iloc[0:-12,0:numbPCs]).reset_index(drop=True)
betaIn3['0'] = np.ones([len(EPFAll.iloc[12:,0]),1])

beta12 = sm.OLS(np.array(EPFAll.iloc[12:,0]),betaIn3, ignore_index=True).fit().params
Y12In = pd.DataFrame(data=TwelveMScore.iloc[:,0:numbPCs])
Y12In['0'] = np.ones([len(TwelveMScore.iloc[:,0]),1])
Y12 = np.dot(Y12In, beta12) 

R12 = np.corrcoef(EPFAll.iloc[12:,0],Y12[0:-12])
plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(EPFAll.iloc[12:,0].reset_index(drop=True))
plt.title('Only 12 month'+ ' PCs' + str(numbPCs) + ' corr: '+ str(R12[0,1]))
plt.plot(Y12)

rotations = [3, 6, 12, 18]

PCAResults = pd.DataFrame(np.ones([len(EPFAll.index),len(rotations)*numbPCs]))
PCANoRotations = pd.DataFrame(np.ones([len(EPFAll.index),len(rotations)*numbPCs]))

for i in range(0,len(rotations)):
    if i == 0:
        PCAResults.iloc[:,i] = np.roll(ThreeMScore.iloc[:,0:numbPCs-1],3)
        PCAResults.iloc[:,i+1] = np.roll(ThreeMScore.iloc[:,1:numbPCs],3)
        PCANoRotations.iloc[:,i] = ThreeMScore.iloc[:,0:numbPCs-1].reset_index(drop=True)
        PCANoRotations.iloc[:,i+1] = ThreeMScore.iloc[:,1:numbPCs].reset_index(drop=True)
    elif i == 1:
        PCAResults.iloc[:,i+1] = np.roll(SixMScore.iloc[:,0:numbPCs-1],6)
        PCAResults.iloc[:,i+2] = np.roll(SixMScore.iloc[:,1:numbPCs],6)
        PCANoRotations.iloc[:,i+1] = SixMScore.iloc[:,0:numbPCs-1].reset_index(drop=True)
        PCANoRotations.iloc[:,i+2] = SixMScore.iloc[:,1:numbPCs].reset_index(drop=True)
    elif i == 2:
        PCAResults.iloc[:,i+2] = np.roll(TwelveMScore.iloc[:,0:numbPCs-1],12)
        PCAResults.iloc[:,i+3] = np.roll(TwelveMScore.iloc[:,1:numbPCs],12)
        PCANoRotations.iloc[:,i+2] = TwelveMScore.iloc[:,0:numbPCs-1].reset_index(drop=True)
        PCANoRotations.iloc[:,i+3] = TwelveMScore.iloc[:,1:numbPCs].reset_index(drop=True)
    elif i == 3:
        PCAResults.iloc[:,i+3] = np.roll(EighteenMScore.iloc[:,0:numbPCs-1],18)
        PCAResults.iloc[:,i+4] = np.roll(EighteenMScore.iloc[:,1:numbPCs],18)
        PCANoRotations.iloc[:,i+3] = EighteenMScore.iloc[:,0:numbPCs-1].reset_index(drop=True)
        PCANoRotations.iloc[:,i+4] = EighteenMScore.iloc[:,1:numbPCs].reset_index(drop=True)

Y = EPFAll.iloc[18:,0].reset_index(drop=True)
const = np.ones([len(Y),1])
resultStorage = pd.DataFrame(data=np.ones([len(EPFAll.index),len(rotations)+1]))

for i in range(1,5):
    #constfull = np.ones([len(SixMScore.iloc[:,0]),1])
    X = pd.DataFrame(data=PCAResults.iloc[18:,((i*numbPCs)-(numbPCs-1))-1:]).reset_index(drop=True)
    X['0'] = const
    Xapp = pd.DataFrame(data=PCAResults.iloc[0:rotations[i-1],((i*numbPCs)-(numbPCs-1)-1):]).reset_index(drop=True)
    Xfull = X.append(Xapp).reset_index(drop=True)
    Xfull = Xfull.fillna(value=1)

    beta = sm.OLS(Y,X, ignore_index=True).fit().params

    Yhat =  np.dot(X,beta)
    residuals = Y - Yhat;
    Yresult = np.dot(Xfull,beta)
    R = np.corrcoef(Y, Yhat)
    resultStorage.iloc[0:len(Yresult),i] = Yresult
    
    plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.plot(Y)
    plt.title('PCA '+ str(rotations[i-1]) + ' PCs' + str(numbPCs) + ' corr: '+ str(R[0,1]))
    plt.plot(Yresult,color='r')

final = resultStorage.set_index(EPFAll.index)
final.to_excel(r'C:\Users\MI2\Desktop\Excel\EPFResults2.xlsx', index = True)