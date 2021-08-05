# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:02:42 2021

@author: MI2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from sklearn.decomposition import PCA
import math
import multiprocessing
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from seriesTransform import *
from fitnetFunction import *
from Norm import *
from columnDrop import *

startDate = dt(2000,1,1) 
endDate_T = dt(2015,12,31)
endDate_N = dt(2021,5,31) 
dataPeriod = 'MONTHLY'
commitmentLevel = 20

mgr = dm.BbgDataManager() 

shadowRateTargetNames = ['FEDL01 Index','WUXIFFRT Index']
    
shadowRateInputNames =[ #Real output and income --- Ix 1:15
        'IP Index', 'IPTLTOTL Index', 'IPTLCG Index',
        'ICGDDCGS Index','IPNDTOTL Index','IPEQBUS Index','IPTLMATS Index',
        'IPMG Index','IPTSOIL Index','INPDRESU Index','CPMFTOT Index',
        'NAPMPMI Index','NAPMPROD Index','PITL Index','COI PI Index',
         #Employment and hours --- Ix 16:39
        'USLFTOT Index','USNATOTL Index','USURTOT Index',
        'USDUMEAN Index','USDULSFV Index','USDUFVFR Index','USDUFIFT Index',
        'USDUFITS Index','NFP T Index','NFP P Index','NFP GP Index','USMMMINE Index',
        'USECTOT Index','USMMMANU Index','USEDTOT Index','USENTOT Index',
        'NFP SP Index','NFP TTUT Index','USEWTOT Index','USRTTOT Index',
        'USEGTOT Index','USWHMANS Index','USWHMNOS Index','NAPMEMPL Index',
         #Consumption --- Ix 40:43
        'PCE CUR$ Index','PCE DRBL Index','PCE NDRB Index','PCE SRV Index',
         #Housing starts and sales --- Ix 44:49
        'NHSPSTOT Index','NHSPSNE Index','NHSPSMW Index','NHSPSSO Index',
        'NHSPSWE Index','NHSPATOT Index', #ln not delta ln in paper
         #Real inventories and orders --- Ix 50:54
        'NAPMINV Index','NAPMNEWO Index','NAPMSUPL Index',
        'LEI NWCN Index','CGNOXAIR Index',
         #Stock prices --- Ix 55:56
        'SPX Index','S5INDU Index',
         #Exchange rates --- Ix 57:58
        'GBP Curncy','CAD Curncy',
         #Interest Rates --- Ix 59:63
        'USGG3M Index','USGG6M Index','USGG2YR Index',
        'USGG5YR Index','USGG10YR Index',
         #Money and credit --- Ix 64:67
        'CCOSNREV Index','M1 Index','M2 Index','FARBAST Index',
         #Price indexes --- Ix 68:81
        'CRB CMDT Index','PPI INDX Index',
        'PPMMTOT Index','PPICTOTL Index','CPI INDX Index','CPSCTOT Index',
        'CPSTTOT Index','CPUMTOT Index','CPUPCXFE Index','CPCADUR Index',
        'CPSSTOT Index','CPUPAXFE Index','CPIQAIFS Index','CPUPAXMC Index',
         #Miscellaneous --- Ix 82:84
        'USHECONN Index','USHEMANN Index','CONSEXP Index']
    
#Inputs sorted by categories with transform switches (delta ln)
transformSwitches = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,  #1:15
                         1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,  #16:39
                         1, 1, 1, 1, #40:43
                         1, 1, 1, 1, 1, 1, #44:49
                         0, 0, 0, 1, 1, #50:54
                         1, 1, #55:56
                         1, 1, #57:58
                         0, 0, 0, 0, 0,  #59:63 #Need to do spreads too
                         1, 1, 1, 1, #64:67
                         0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  #68:81
                         1, 1, 0]#82:84
    
# Data Collection and Cleaning
# Retreiving Bloomberg data and transforming as per Wu Xia
sids1 = mgr[shadowRateInputNames]
sRIAll = sids1.get_historical(['PX_LAST'],startDate,endDate_N, dataPeriod)
sRIAll.columns = sRIAll.columns.droplevel(1)
sRIAll = sRIAll[shadowRateInputNames]
sRIAll = columnDrop(sRIAll)
sRIDates = sRIAll.index
#print(str(sRIDates[0]))
#print(str(sRIDates[-1]))
sRIAll = seriesTransform(sRIAll,transformSwitches)
sids2 = mgr[shadowRateTargetNames]
sRTAll = sids2.get_historical(['PX_LAST'],startDate,endDate_T, dataPeriod)
sRTAll.columns = sRTAll.columns.droplevel(1)
sRTAll = sRTAll[shadowRateTargetNames]
sRTAll = columnDrop(sRTAll)
sRTDates = sRTAll.index
#print(sRTDates[1])
#print(sRTDates[-1])
for i in range(0,len(sRTDates)):
    if sRIDates[i] == sRTDates[-1]:
        lastDateofShadowIndex = i
    
#Create bond spreads and include in data
for i in range(1,5):
    value = sRIAll.iloc[:,58+i].sub(sRIAll.iloc[:,58])
    sRIAll[str(len(sRIAll)+i)] = value
print('Finished Acquisition and Cleaning')

# ANN Traning on Everything
#Parameters
tTimes = 10*commitmentLevel
hiddenLayerSize = 20
inputs = sRIAll.iloc[1:lastDateofShadowIndex+1,:]
targets = sRTAll.iloc[1:,1]
#mdlStorage = pd.DataFrame(np.zeros([tTimes, len(sRIAll.iloc[1:,0])]))

num_cores = multiprocessing.cpu_count()
results1 = (Parallel(n_jobs=num_cores, verbose=50)(delayed(fitnetFunction)(inputs, targets, sRIAll.iloc[1:,:],20)for i in range(tTimes)))
#BLIP AT 242
# plot initial ANN
mdlResult = np.mean(np.array(results1), axis=0)
#mdlResDv = np.std(md1Result)
#mdl2up = mdlResult[:].add(2 * mdlResDv[:])
#mdl2down = mdlResult[:].sub(2 * mdlResDv[:])
#mdlSample = results[math.ceil(tTimes/2),:]

plt.figure(figsize=[15,10])
plt.grid(True)    
plt.plot(sRTAll.iloc[1:,0],color='g')
plt.title('All Inputs')
plt.plot(sRTAll.iloc[1:,1],color='b')
plt.plot(mdlResult, color='r')
#plt.plot(mdlSample, color='k')
#plt.plot(mdl2up,color='c')
#plt.plot (mdl2down, color='y')
plt.show()
# Major Component ANN
# Input names
# Categories are major factors listed in Wu Xia
shadowRateMajorInputs = [ #Real output and income --- Ix 1:15
        'IP Index', 'IPTLTOTL Index', 'IPTLCG Index',
        'ICGDDCGS Index','IPNDTOTL Index','IPEQBUS Index','IPTLMATS Index',
        'IPMG Index','IPTSOIL Index','INPDRESU Index','CPMFTOT Index',
        'NAPMPMI Index','NAPMPROD Index','PITL Index','COI PI Index',
         #Employment and hours --- Ix 16:39
        'USLFTOT Index','USNATOTL Index','USURTOT Index',
        'USDUMEAN Index','USDULSFV Index','USDUFVFR Index','USDUFIFT Index',
        'USDUFITS Index','NFP T Index','NFP P Index','NFP GP Index','USMMMINE Index',
        'USECTOT Index','USMMMANU Index','USEDTOT Index','USENTOT Index',
        'NFP SP Index','NFP TTUT Index','USEWTOT Index','USRTTOT Index',
        'USEGTOT Index','USWHMANS Index','USWHMNOS Index','NAPMEMPL Index',
         #Price indexes --- Ix 40:53
        'CRB CMDT Index','PPI INDX Index',
        'PPMMTOT Index','PPICTOTL Index','CPI INDX Index','CPSCTOT Index',
        'CPSTTOT Index','CPUMTOT Index','CPUPCXFE Index','CPCADUR Index',
        'CPSSTOT Index','CPUPAXFE Index','CPIQAIFS Index','CPUPAXMC Index'];
majorTransformSwitches = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
# Data Collecting and Cleaning
sids3 = mgr[shadowRateMajorInputs]
sRMIAll = sids3.get_historical(['PX_LAST'],startDate, endDate_N, dataPeriod)
sRMIAll.columns = sRMIAll.columns.droplevel(1)
sRMIAll = sRMIAll[shadowRateMajorInputs]
sRMIAll = columnDrop(sRMIAll)
sRMIDates = sRMIAll.index
#print(str(sRMIDates[0]))
#print(str(IsRMIDates[-1]))
sRMIAll = seriesTransform(sRMIAll,majorTransformSwitches)
#sRMIAll = normal_cal(sRMIAll)
for i in range(0,len(sRTDates)):
    if sRMIDates[i] == sRTDates[-1]:
        lastDateofShadowIndexMajor = i

# ANN
#Parameters
tTimesM = 10*commitmentLevel;
hiddenLayerSizeM = 20
inputsM = sRMIAll.iloc[1:lastDateofShadowIndexMajor+1,:]
targetsM = sRTAll.iloc[1:,1]
#mdlStorageM = pd.DataFrame(np.zeros([tTimesM, len(sRMIAll.iloc[1:,0])]))
#bestEpochStorage = pd.DataFrame(np.zeros([tTimesM,1]))
        
num_cores = multiprocessing.cpu_count()
results2 = (Parallel(n_jobs=num_cores, verbose=50)(delayed(fitnetFunction)(inputsM, targetsM, sRMIAll.iloc[1:,:],15)for i in range(tTimes)))
#BLIP AT 242. INPUT DF IS SAME AS MATLAB
# plot Component ANN
mdlResultM = np.mean(np.array(results2), axis=0)
#mdlResDvM = np.std(mdlStorageM)
#mdl2upM = mdlResultM[:].add(2 * mdlResDvM[:])
#mdl2downM = mdlResultM[:].sub(2 * mdlResDvM[:])
 
plt.figure(figsize=[15,10])
plt.grid(True)   
plt.plot(sRTAll.iloc[1:,0],color='g')
plt.title('Major Components')
plt.plot(sRTAll.iloc[1:,1],color='b')
plt.plot(mdlResultM, color='r')
#plt.plot(mdl2upM,color='c')
#plt.plot (mdl2downM, color='y')
plt.show()

# All Input PCA ANN
nPCs = 2
pca = PCA(n_components=len(sRIAll.columns))
principalComponents1 = pca.fit_transform(columnDrop(sRIAll))
sRScore = pd.DataFrame(data=principalComponents1).reset_index(drop=True)
#Parameters
tTimesP = 10*commitmentLevel
hiddenLayerSizeP = 20
inputsP = sRScore.iloc[1:lastDateofShadowIndexMajor+1,0:nPCs]
targetsP = sRTAll.iloc[1:,1]
#mdlStorageP = pd.DataFrame(np.zeros([tTimesP, len(sRScore.iloc[1:,0])]))

num_cores = multiprocessing.cpu_count()
results3 = (Parallel(n_jobs=num_cores, verbose=50)(delayed(fitnetFunction)(inputsP, targetsP, sRScore.iloc[:,0:nPCs],15)for i in range(tTimes)))
        
# plot All Input PCA ANN
mdlResultP = np.mean(np.array(results3), axis=0)
#mdlResDvP = np.std(mdlStorageP)
#mdl2upP = mdlResultP[:].add(2 * mdlResDvP[:])
#mdl2downP = mdlResultP[:].sub(2 * mdlResDvP[:])
   
plt.figure(figsize=[15,10])
plt.grid(True) 
plt.plot(sRTAll.iloc[1:,0],color='g')
plt.title('All Inputs PCA')
plt.plot(sRTAll.iloc[1:,1],color='b')
plt.plot(mdlResultP, color='r')
#plt.plot(mdl2upP,'c')
#plt.plot (mdl2downP, 'y')
plt.show()

# Major Component PCA ANN
nMPCs = 2
pca = PCA(n_components=len(sRMIAll.columns))
principalComponents1 = pca.fit_transform(columnDrop(sRMIAll))
sRMScore = pd.DataFrame(data=principalComponents1).reset_index(drop=True) 
sRMCoeff = pd.DataFrame(pca.components_).transpose()
plt.figure(figsize=[15,10])
plt.grid(True) 
plt.plot(sRMCoeff.iloc[1,:])
plt.show()
#Parameters
tTimesMP = 10*commitmentLevel
hiddenLayerSizeMP = 25
inputsMP = sRMScore.iloc[1:lastDateofShadowIndexMajor+1,0:nMPCs]
targetsMP = sRTAll.iloc[1:,1]
#mdlStorageMP = pd.DataFrame(np.zeros([tTimesMP, len(sRMIAll.iloc[1:,0])]))
        
num_cores = multiprocessing.cpu_count()
results4 = (Parallel(n_jobs=num_cores, verbose=50)(delayed(fitnetFunction)(inputsMP, targetsMP, sRMScore.iloc[:,0:nPCs],15)for i in range(tTimes)))
        
# plot Major Component PCA ANN
mdlResultMP = np.mean(np.array(results4), axis=0)
#mdlResDvMP = np.std(mdlStorageMP)
#mdl2upMP = mdlResultMP[:].add(2 * mdlResDvMP[:])
#mdl2downMP = mdlResultMP[:].sub(2 * mdlResDvMP[:])

plt.figure(figsize=[15,10])
plt.grid(True)    
plt.plot(sRTAll.iloc[1:,0], color='g')
plt.title('Major Components PCA ' + str(nMPCs))
plt.plot(sRTAll.iloc[1:,1],color='b')
plt.plot(mdlResultMP, color='r')
#plt.plot(mdlResultP, 'k')
#plt.plot(mdl2upP,'c')
#plt.plot (mdl2downP, 'y')
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(sRMScore.iloc[1:,0:nMPCs])
plt.show()

#export1 = pd.DataFrame(mdlResultMP).set_index(sRMIDates[2:])
#export2 = pd.DataFrame(mdlResult).set_index(sRMIDates[1:])
#export1.to_excel(r'C:\Users\MI2\Desktop\Excel\ShadowRateMajorPCATrunc2.xlsx', index = True)
#export2.to_excel(r'C:\Users\MI2\Desktop\Excel\ShadowRatePCATrunc.xlsx', index = True)