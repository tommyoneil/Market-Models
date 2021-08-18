# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:15:23 2021

@author: MI2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tia.bbg.datamgr as dm
from datetime import date as dt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import math
import multiprocessing
from joblib import Parallel, delayed
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from seriesTransform import *
from fitnetFunction import *
from Norm import *
from gaussian import *
from yoyMonthAvg import *
import warnings

warnings.filterwarnings("ignore")
    
nStartI = dt(2002,1,1)
nStartO = dt(2001,2,1)
# nStartI = dt(2011,1,1)
# nStartO = dt(2010,2,1)
mgr = dm.BbgDataManager() 
n = dt(2021,6,30)
num_cores = multiprocessing.cpu_count()
dataPeriod = 'DAILY'
## =======================================================================
# Pull inflation and oil data, inflation is current, while oil goes out a
# year using monthly average of YoY using current  
yearOutOilChange = 0 #Percentage

#inflationTicker = 'GRCP2HYY Index'
inflationTicker = 'CPI YOY Index'
#inflationTicker = 'CACPIYOY Index'
#inflationTicker = 'ECCPEMUY Index'
#inflationTicker = 'EUPPEMUY Index'
#inflationTicker = 'FRCPEECY Index'
#inflationTicker = 'FDIUFDYO Index'
#inflationTicker = 'PPI YOY Index'


#comdtyTicker = 'CL6 Comdty'
#comdtyTicker = 'SPGSIN Index'
#comdtyTicker = 'TWI EUSP Index'
comdtyTicker = 'EUCRBRDT Index'
#comdtyTicker = 'CO4 Comdty'
#comdtyTicker = 'EURUSD Curncy'
#comdtyTicker = 'USTWBROA Index'
#comdtyTicker = 'USCRWTIC Index'

sids1 = mgr[comdtyTicker]
comdtyIdxRaw = sids1.get_historical(['PX_LAST'],nStartO,n, dataPeriod)
#comdtyIdxRaw.columns = comdtyIdxRaw.columns.droplevel(1)
        
comdtyIdx = yoyMonthAvg(comdtyIdxRaw,250,yearOutOilChange)
sids2 = mgr[inflationTicker]
inflationIdx = sids2.get_historical(['PX_LAST'],nStartI,n, dataPeriod)


## =======================================================================
# Quick check that we have the right number of data points and that there
# are no issues due to non-identical  of month dates. i.e. Dec 30 vs 31
print('Oil Data Start ',str(dt.fromordinal(int(comdtyIdx.iloc[0,0]))))
print('Inflation Data Start ', str(inflationIdx.index[0]))
print('Oil Data Comp  ',str(dt.fromordinal(int(comdtyIdx.iloc[len(inflationIdx.index)-1,0]))))
print('Inflation Data  ',str(inflationIdx.index[-1]))
print('Oil Data  ',str(dt.fromordinal(int(comdtyIdx.iloc[-1,0]))))


## =======================================================================
# Begin functional code, repititions and maxLag are self-explanatory
maxLag = 6
fitFraction = 0.7
commitmentLevel = 60
repititions = int(commitmentLevel / 5)
sectionNames = ['MSEin','MSEfinal','MSEcomp']
modelNames = ['OLSHAC','OLSHAC all subsets','Lasso','SeqFS','ANNFS','Broad ANN']
modelSelected = 'ANNFS'


## ===================================================================
# Xfull is yoy oil data with columns from concurrent to lagged maxLag months
#   includes rotated rows that contain wrapped data (need to be
#   eliminated for final fit)
# Y is inflation data, renamed for simplicity, excluding maxLag
# Xcomp is yoyoil for the time period matching Y
# Yin is data selected to be in Fitting
# Yfinal is data selected for Model Analysis
# Xin is data selected to be in Fitting
# Xfinal is data slected for Model Analysis
MSEin = []
MSEfinal = []
MSEcomp = []
fullstorage = []
FSstorage = []
for reps in range(repititions):
    XfullPrep = [list(comdtyIdx.iloc[:,1]), list(np.roll(comdtyIdx.iloc[:,1],1)), list(np.roll(comdtyIdx.iloc[:,1],2)),
            list(np.roll(comdtyIdx.iloc[:,1],3)), list(np.roll(comdtyIdx.iloc[:,1],4)), list(np.roll(comdtyIdx.iloc[:,1],5)),
            list(np.roll(comdtyIdx.iloc[:,1],6))] #, list(np.roll(comdtyIdx.iloc[:,1],7)), list(np.roll(comdtyIdx.iloc[:,1],8)),
        #             list(np.roll(comdtyIdx.iloc[:,1],12)), list(np.roll(comdtyIdx.iloc[:,1],16)), list(np.roll(comdtyIdx.iloc[:,1],24)),
        #             list(np.roll(comdtyIdx.iloc[:,1],30))])
    Xfull = pd.DataFrame(XfullPrep).transpose()
    
    Y = pd.DataFrame(inflationIdx.iloc[maxLag:,0])
    Xcomp = pd.DataFrame(Xfull.iloc[maxLag:maxLag+len(Y),:])
    binaryVector1 = np.ones([1,round(len(Y)*fitFraction)])
    addOn = np.zeros([1,len(Y)-round(len(Y)*fitFraction)])
    binaryVector = list(np.append(binaryVector1,addOn))
    inSampleVector = [int(i) for i in np.random.permutation(binaryVector)]
    YinPrep = []
    for i in range(0,len(inSampleVector)):
        if inSampleVector[i] == 1:
            YinPrep.append(Y.iloc[i,0])
        else:
            continue
    Yin = pd.DataFrame(YinPrep)
    
    YfinalPrep = []
    for i in range(0,len(inSampleVector)):
        if inSampleVector[i] == 0:
            YfinalPrep.append(Y.iloc[i,0])
        else:
            continue
    Yfinal = pd.DataFrame(YfinalPrep)
    
    colList = []
    for j in range(0,len(Xcomp.columns)):
        numList = []
        for i in range(0,len(inSampleVector)):
            if inSampleVector[i] == 1:
                    numList.append(Xcomp.iloc[i,j])
            else:
                continue
        colList.append(numList)
    Xin = pd.DataFrame(colList).transpose()
        
    colList2 = []
    for j in range(0,len(Xcomp.columns)):
        numList2 = []
        for i in range(0,len(inSampleVector)):
            if inSampleVector[i] == 0:
                    numList2.append(Xcomp.iloc[i,j])
            else:
                continue
        colList2.append(numList2)
        
    Xfinal = pd.DataFrame(colList2).transpose()
                    
    Ximportant = Xfull.iloc[maxLag:,:]
    importantDates = comdtyIdx.iloc[maxLag:,0]
    
    # Use Fitting Data in a variety of models
    # No constant
    olsBeta = sm.OLS(Yin,Xin).fit().params
    Yols = np.dot(Xin,olsBeta)
    Yolsfinal = np.dot(Xfinal,olsBeta)
    Yolscomp = np.dot(Xcomp,olsBeta)
    # With constant
    Xin1 = pd.DataFrame(Xin.copy())
    Xin1['1'] = np.ones([len(Xin),1])
    olsConstantBeta = sm.OLS(Yin,Xin1).fit().params
    YolsConstant = np.dot(Xin1,olsConstantBeta)
    Xfinal1 = pd.DataFrame(Xfinal.copy())
    Xfinal1['1'] = np.ones([len(Xfinal),1])
    YolsConstantfinal = np.dot(Xfinal1,olsConstantBeta)
    Xcomp1 = pd.DataFrame(Xcomp.copy())
    Xcomp1['1'] = np.ones([len(Xcomp),1])
    YolsConstantcomp = np.dot(Xcomp1,olsConstantBeta)
    # plt.plot(results)
    # plt.plot(Yin,color='g')
    # plt.title('OLSHAC')
    # plt.plot(Yols,color='b') 
    # plt.plot(YolsConstant,color='k')
    # plt.show()
    
    # Optimal OLSHAC from All Possible Subsets
    subSets = [bin(i).replace('0b','') for i in range(1,(2**(len(Xin.columns))))]
    newSubSets = []
    for i in subSets:
        if len(i) != len(Xin.columns):
            diff = len(Xin.columns) - len(i)
            diffList = []
            for j in range(0,diff):
                diffList.append('0')
            diffList.append(i)
            newStr = ''.join(diffList)
            newNum = i.replace(i,newStr)
            newSubSets.append(newNum)
        else:
            newSubSets.append(i)
            
    datList = []
    for i in newSubSets:
        charList = []
        for j in range(0,len(i)):
            charList.append(i[j])
        datList.append(charList)
    subSets = pd.DataFrame(datList)
         
    allSetResults = []
    for setNumber in range(0,len(subSets)):
        setLogical = [int(i) for i in subSets.iloc[setNumber,:]]
        Xcross1 = Xin.copy()
        for i in range(0,len(Xcross1.columns)):
            if setLogical[i] == 0:
                Xcross1 = Xcross1.drop([i], axis=1)
            else:
                continue
        Xcross = Xcross1.copy()
    
        model = svm.SVC() #kernel='linear'
        Ycross = [int(i) for i in np.array(Yin)]
        allSetResults.append(np.mean(abs((cross_val_score(model, Xcross, Ycross, scoring = 'neg_mean_squared_error', cv = 5)))))
    
    subSets['MSE'] = allSetResults
    subSets = subSets.sort_values('MSE')
    
    # No constant
    XinCheckPrep = []
    checker = list(subSets.iloc[0,0:-1])
    for i in range(0,len(checker)):
        if checker[i] == str(1):
            XinCheckPrep.append(Xin.iloc[:,i])
        else:
            continue
    XinCheck = pd.DataFrame(XinCheckPrep).transpose()
    bestSetBeta = np.array(sm.OLS(Yin,XinCheck).fit().params)
    
    bestSetHolder = []
    checkBeta = bestSetBeta.copy()
    for i in range(0,len(Xin.iloc[0,:])):
        if checker[i] == str(1):
            bestSetHolder.append(checkBeta[0])
            np.delete(checkBeta,0)
        else:
            bestSetHolder.append(0)
        
    YbestSet = np.dot(Xin,bestSetHolder)
    YbestSetfinal = np.dot(Xfinal,bestSetHolder)
    YbestSetcomp = np.dot(Xcomp,bestSetHolder)
    
    # With constant
    XinAlt = XinCheck.copy()
    XinAlt['1'] = np.ones([len(Xin.iloc[:,0]),1])
    bestSetConstantBeta = sm.OLS(Yin,XinAlt).fit().params
    
    YbestSetConstant = np.dot(XinAlt,bestSetConstantBeta)
    
    XfinalCheckPrep = []
    checker = list(subSets.iloc[0,0:-1])
    for i in range(0,len(checker)):
        if checker[i] == str(1):
            XfinalCheckPrep.append(Xfinal.iloc[:,i])
        else:
            continue
        
    XfinalCheck = pd.DataFrame(XfinalCheckPrep).transpose()
    XfinalAlt = XfinalCheck.copy()
    XfinalAlt['1'] = np.ones([len(Xfinal.iloc[:,0]),1])
    
    YbestSetConstantfinal = np.dot(XfinalAlt,bestSetConstantBeta)
    
    XcompCheckPrep = []
    checker = list(subSets.iloc[0,0:-1])
    for i in range(0,len(checker)):
        if checker[i] == str(1):
            XcompCheckPrep.append(Xcomp.iloc[:,i])
        else:
            continue
        
    XcompCheck = pd.DataFrame(XcompCheckPrep).transpose()
    XcompAlt = pd.DataFrame(XcompCheck)
    XcompAlt['1'] = np.ones([len(Xcomp.iloc[:,0]),1])
    
    YbestSetConstantcomp = np.dot(XcompAlt,bestSetConstantBeta)
    # plt.plot(results)
    # plt.plot(Yin,color='g')
    # plt.title('Best Set')
    # plt.plot(YbestSet,color='r')
    # plt.plot(YbestSetConstant,color='k')
    # plt.show()
    
    # Lasso
    # No constant   
    X_train, X_test, y_train, y_test = train_test_split(Xin, Yin)
    reg = LassoCV(cv=5).fit(X_train, y_train)
    lassoBeta = reg.coef_
    
    lassoBeta = pd.DataFrame(lassoBeta)
    
    Ylasso = np.dot(Xin,lassoBeta)
    Ylassofinal = np.dot(Xfinal,lassoBeta)
    Ylassocomp = np.dot(Xcomp,lassoBeta)
    Ylassoimportant = np.dot(Ximportant,lassoBeta)
    # With constant (not indepent lasso bc can't include
    # constant)
    
    Ylasso1 = pd.DataFrame(Ylasso)
    Ylasso1['1'] = np.ones([len(Ylasso),1])
    lassoConstantBeta = sm.OLS(Yin,Ylasso1).fit().params
    YlassoWithConstant = np.dot(Ylasso1,lassoConstantBeta)
    Ylassofinal1 = pd.DataFrame(Ylassofinal)
    Ylassofinal1['1'] = np.ones([len(Ylassofinal),1])
    YlassoWithConstantfinal = np.dot(Ylassofinal1,lassoConstantBeta)
    Ylassocomp1 = pd.DataFrame(Ylassocomp)
    Ylassocomp1['1'] = np.ones([len(Ylassocomp),1])
    YlassoWithConstantcomp = np.dot(Ylassocomp1,lassoConstantBeta)
    Ylassoimportant1 = pd.DataFrame(Ylassoimportant)
    Ylassoimportant1['1'] = np.ones([len(Ylassoimportant),1])
    YlassoWithConstantimportant = np.dot(Ylassoimportant1,lassoConstantBeta)
    if modelSelected == 'Lasso':
        fullstorage.append(YlassoWithConstantimportant)#LASSO MDL
    
    # plt.plot(results)
    # plt.plot(Yin,color='g')
    # plt.title('Lasso results')
    # plt.plot(Ylasso, color='r')
    # plt.plot(YlassoWithConstant,color='k')
    # plt.show()
    
    # Sequential Feature Selection
    samplingRounds = commitmentLevel * 2
    sumin = []
    for j in range(0,samplingRounds):
        X_train, X_test, y_train, y_test = train_test_split(Xin, Yin)
        def RSS(Xin, Yin): #RESIDUAL SUM CALC
            import statsmodels.api as sm
            x = sm.add_constant(Xin)
            model = sm.OLS(Yin, Xin).fit().ssr#NEED TO MATCH RESIDUAL SUM OF SQUARES SCORING METHOD  OR USE NEG MEAN SQ ERR ARE THEY SAME?
        estimator = LinearRegression().fit(X_train, y_train)
        sfs_selector = SequentialFeatureSelector(estimator, scoring='neg_mean_squared_error', cv=5)#scoring=RSS(Xin, Yin), cv=5)
        sfs_selector.fit(X_train, y_train)
        insample = sfs_selector.get_support() #GETS SAME FEATURES EACH ITERATION!!! THATS WRONG
        print(insample)
        inAdd = []
        for i in insample:
            if i == True:
                inAdd.append(1)
            else:
                inAdd.append(0)
        sumin.append(inAdd)
        print(j)
    sumin = pd.DataFrame(sumin).sum(axis=0)
    print('Done FS')
    
    # printlay histogram of feature inclusion over all sampling
    # Rounds
    realVector = []
    for i in range(0,len(sumin)):# Threshold Arbitrarily set to 50# of samples
        if sumin[i] > (samplingRounds/5):
            realVector.append(str(1))
        else:
            realVector.append(str(0))
    
    FSstorage.append(sumin)
    # No constant
    FSBetaCheck1 = []
    for i in range(0,len(realVector)):
        if realVector[i] == str(1):
            FSBetaCheck1.append(Xin.iloc[:,i])
        else:
            continue
    FSBetaCheck = pd.DataFrame(FSBetaCheck1).transpose()
    FSBeta = sm.OLS(Yin,FSBetaCheck).fit().params
    YFS = np.dot(FSBetaCheck,FSBeta)
    FSBetaCheck2 = []
    for i in range(0,len(realVector)):
        if realVector[i] == str(1):
            FSBetaCheck2.append(Xfinal.iloc[:,i])
        else:
            continue
        
    FSBetaCheck12 = pd.DataFrame(FSBetaCheck2).transpose()
    YFSfinal = np.dot(FSBetaCheck12,FSBeta)
    FSBetaCheck3 = []
    for i in range(0,len(realVector)):
        if realVector[i] == str(1):
            FSBetaCheck3.append(Xcomp.iloc[:,i])
        else:
            continue
    FSBetaCheck13 = pd.DataFrame(FSBetaCheck3).transpose()
    
    FSBetaCheck4 = []
    for i in range(0,len(realVector)):
        if realVector[i] == str(1):
            FSBetaCheck4.append(Ximportant.iloc[:,i])
        else:
            continue
    FSBetaCheck14 = pd.DataFrame(FSBetaCheck4).transpose()
    
    YFScomp = np.dot(FSBetaCheck13,FSBeta)
    # With constant (not indepent FS bc can't include
    # constant)
    Xin2 = FSBetaCheck.copy()
    Xin2['1'] = np.ones([len(Xin.iloc[:,0]),1])
    FSConstantBeta = sm.OLS(Yin,Xin2).fit().params
    YFSConstant = np.dot(Xin2,FSConstantBeta)
    Xfinal2 = FSBetaCheck12.copy()
    Xfinal2['1'] = np.ones([len(Xfinal.iloc[:,0]),1])
    YFSConstantfinal = np.dot(Xfinal2,FSConstantBeta)
    Xcomp1 =  FSBetaCheck13.copy()
    Xcomp1['1'] = np.ones([len(Xcomp.iloc[:,0]),1])
    YFSConstantcomp = np.dot(Xcomp1,FSConstantBeta)
    # plt.plot(results)
    # plt.plot(Yin,color='g')
    # plt.title('Feature Selection')
    # plt.plot(YFS,color='r')
    # plt.plot(YFSConstant,color='k')
    # plt.show()
    
    # ANN from Selected Features
    #Parameters
    tTimesFS = commitmentLevel * 20
    hiddenLayerSizeFS = 20
    inputsFS = FSBetaCheck.copy()
    targetsFS = Yin.copy()
    
    #Run ANN
    def ANNFS1():
        mdlStorageFS = fitnetFunction(inputsFS, targetsFS, FSBetaCheck, hiddenLayerSizeFS)
        mdlStorageFSfinal = fitnetFunction(inputsFS, targetsFS, FSBetaCheck12, hiddenLayerSizeFS)
        mdlStorageFScomp = fitnetFunction(inputsFS, targetsFS, FSBetaCheck13, hiddenLayerSizeFS)
        mdlStorageFSimportant = fitnetFunction(inputsFS, targetsFS, FSBetaCheck14, hiddenLayerSizeFS)      
        return [mdlStorageFS, mdlStorageFSfinal, mdlStorageFScomp, mdlStorageFSimportant]
    
    results = (Parallel(n_jobs=num_cores, verbose=True)(delayed(ANNFS1)()for i in range(0,tTimesFS))) 
    
    mdlStorageFS = []
    mdlStorageFSfinal = []
    mdlStorageFScomp = []
    mdlStorageFSimportant = []
    
    for i in range(0,len(results)):
        for j in range(0,len(results[i])):
            if j == 0:
                mdlStorageFS.append(results[i][j])
            elif j == 1:
                mdlStorageFSfinal.append(results[i][j])
            elif j == 2:
                mdlStorageFScomp.append(results[i][j])
            elif j == 3:
                mdlStorageFSimportant.append(results[i][j])
                
    mdlStorageFS = pd.DataFrame(mdlStorageFS)
    mdlStorageFSfinal = pd.DataFrame(mdlStorageFSfinal)
    mdlStorageFScomp = pd.DataFrame(mdlStorageFScomp)
    mdlStorageFSimportant = pd.DataFrame(mdlStorageFSimportant)
    
    print('Done ANN on FS')
    # ANN Results
    YANNFS = mdlStorageFS.mean(axis=0)
    YANNFSfinal = mdlStorageFSfinal.mean(axis=0)
    YANNFScomp = mdlStorageFScomp.mean(axis=0)
    YANNFSimportant = mdlStorageFSimportant.mean(axis=0)
    
    if modelSelected == 'ANNFS':
        fullstorage.append(YANNFSimportant)#ANNFS MDL
    
    #fullstorage.append(YANNFSimportant) # TURN ON TO USE ANNFS
    #FOR OVERALL MODEL
    # Sample extraction for visualization
    mdlSample1FS = mdlStorageFS.iloc[math.ceil(tTimesFS/2),:]
    mdlSample2FS = mdlStorageFS.iloc[math.ceil(tTimesFS/3),:]
    mdlSample3FS = mdlStorageFS.iloc[math.ceil(tTimesFS/4),:]
    mdlSample4FS = mdlStorageFS.iloc[math.ceil(tTimesFS/5),:]
    # Individual model analysis and selection
    ANNresultsFS = pd.DataFrame(mdlStorageFS.copy())
    addCol1 = []
    addCol2 = []
    for i in range(0,len(mdlStorageFS.iloc[:,0])):
        mdin1 = np.array(mdlStorageFS.iloc[i,:])
        gutsFS = pd.DataFrame(np.corrcoef(Yin.copy().transpose(),mdin1))
        addCol1.append(gutsFS.iloc[0,1])
        Ycross1 = [int(i) for i in np.array(Yin)]
        model = svm.SVC() #kernel='linear'
        crossValues = abs((cross_val_score(model, pd.DataFrame(mdin1), Ycross1, scoring = 'neg_mean_squared_error', cv = 5)))
        addCol2.append(np.mean(crossValues))
    ANNresultsFS['Corr'] = addCol1
    ANNresultsFS['CV Mean'] = addCol2
    
    bestCorrsFS = ANNresultsFS.sort_values(by=['Corr'], axis=0)
    bestCorrFS = bestCorrsFS.iloc[-1,0:-1]
    bestMSEsFS = ANNresultsFS.sort_values(by=['CV Mean'], axis=0)
    bestMSEFS = bestMSEsFS.iloc[0,0:-1]
    # plt.plot(results)
    # plt.plot(Yin,color='g')
    # plt.title('ANN FS results')
    # plt.plot(YANNFS,color= 'k')
    # plt.plot(bestCorrFS,color= 'r')
    # plt.plot(bestMSEFS, color='b')
    # plt.show()
    
    # Broad ANN
    #Parameters
    tTimes = commitmentLevel * 20
    hiddenLayerSize = 20
    inputs = Xin.copy()
    targets = Yin.copy()
    
    #Run ANN
    def ANNFS2():
        mdlStorage = fitnetFunction(inputs, targets, Xin, hiddenLayerSize)
        mdlStoragefinal = fitnetFunction(inputs, targets, Xfinal, hiddenLayerSize)
        mdlStoragecomp = fitnetFunction(inputs, targets, Xcomp, hiddenLayerSize)
        mdlStorageimportant = fitnetFunction(inputs, targets, Ximportant, hiddenLayerSize)      
        return [mdlStorage, mdlStoragefinal, mdlStoragecomp, mdlStorageimportant]
    
    results2 = (Parallel(n_jobs=num_cores, verbose=True)(delayed(ANNFS2)()for i in range(0,tTimes))) 
    
    mdlStorage = []
    mdlStoragefinal = []
    mdlStoragecomp = []
    mdlStorageimportant = []
    
    for i in range(0,len(results2)):
        for j in range(0,len(results2[i])):
            if j == 0:
                mdlStorage.append(results2[i][j])
            elif j == 1:
                mdlStoragefinal.append(results2[i][j])
            elif j == 2:
                mdlStoragecomp.append(results2[i][j])
            elif j == 3:
                mdlStorageimportant.append(results2[i][j])
                
    mdlStorage = pd.DataFrame(mdlStorage)
    mdlStoragefinal = pd.DataFrame(mdlStoragefinal)
    mdlStoragecomp = pd.DataFrame(mdlStoragecomp)
    mdlStorageimportant = pd.DataFrame(mdlStorageimportant)
    print('Done ANN on Everything')
    # ANN results
    YANN = mdlStorage.mean(axis=0)
    YANNfinal = mdlStoragefinal.mean(axis=0)
    YANNcomp = mdlStoragecomp.mean(axis=0)
    YANNimportant = mdlStorageimportant.mean(axis=0)
    if modelSelected == 'Broad ANN':
        fullstorage.append(YANNimportant)#Broad ANN MDL
    
    #fullstorage.append(YANNimportant) # TURN ON TO USE ANN
    # Sample extraction for visualization
    mdlSample1 = mdlStorage.iloc[math.ceil(tTimes/2),:]
    mdlSample2 = mdlStorage.iloc[math.ceil(tTimes/3),:]
    mdlSample3 = mdlStorage.iloc[math.ceil(tTimes/4),:]
    mdlSample4 = mdlStorage.iloc[math.ceil(tTimes/5),:]
    # Individual model analysis and selection
    ANNresults = pd.DataFrame(mdlStorage.copy())
    addCol3 = []
    addCol4 = []
    for i in range(0,len(mdlStorage.iloc[:,0])):
        mdin = np.array(mdlStorage.iloc[i,:])
        guts = pd.DataFrame(np.corrcoef(Yin.copy().transpose(),mdin))
        addCol3.append(guts.iloc[0,1])
        Ycross2 = [int(i) for i in np.array(Yin)]
        model = svm.SVC() #kernel='linear'
        crossValues2 = abs((cross_val_score(model, pd.DataFrame(mdin), Ycross2, scoring = 'neg_mean_squared_error', cv = 5)))
        addCol4.append(np.mean(crossValues2))
    ANNresults['Corr'] = addCol3
    ANNresults['CV Mean'] = addCol4
    
    bestCorrs = ANNresults.sort_values(by=['Corr'], axis=0)
    bestCorr = bestCorrs.iloc[-1,0:-1]
    bestMSEs = ANNresults.sort_values(by=['CV Mean'], axis=0)
    bestMSE = bestMSEs.iloc[0,0:-1]
    # plt.plot(results)
    # plt.plot((Y),color='g')
    # plt.title('ANN results')
    # plt.plot(YANNcomp,color= 'k')
    # plt.plot(bestCorr,color= 'r')
    # plt.plot(bestMSE, color='b')
    # plt.show()
    
    # Evaluate MSE of all resuls
    # Insample vs out of samples versus full comparable
    MSEin.append([mean_squared_error(YolsConstant,Yin), mean_squared_error(YbestSetConstant,Yin), mean_squared_error(YlassoWithConstant,Yin), mean_squared_error(YFSConstant,Yin),
    mean_squared_error(YANNFS,Yin), mean_squared_error(YANN,Yin)])
    MSEfinal.append([mean_squared_error(YolsConstantfinal,Yfinal), mean_squared_error(YbestSetConstantfinal,Yfinal), mean_squared_error(YlassoWithConstantfinal,Yfinal),
    mean_squared_error(YFSConstantfinal,Yfinal), mean_squared_error(YANNFSfinal,Yfinal), mean_squared_error(YANNfinal,Yfinal)])
    MSEcomp.append([mean_squared_error(YolsConstantcomp,Y), mean_squared_error(YbestSetConstantcomp,Y), mean_squared_error(YlassoWithConstantcomp,Y),
    mean_squared_error(YFSConstantcomp,Y), mean_squared_error(YANNFScomp,Y), mean_squared_error(YANNcomp,Y)])
    
    print(str(reps+1), ' repitition completed!')

        
MSEin = pd.DataFrame(MSEin)
MSEfinal = pd.DataFrame(MSEfinal)
MSEcomp = pd.DataFrame(MSEcomp)
fullstorage = pd.DataFrame(fullstorage)
FSstorage = pd.DataFrame(FSstorage)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(MSEin)
plt.legend(modelNames)
plt.title('MSEin')
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(MSEfinal)
plt.legend(modelNames)
plt.title('MSEfinal')
plt.show()

errorStorage = [np.array(MSEin.sum(axis=0)), np.array(MSEfinal.sum(axis=0)), np.array(MSEcomp.sum(axis=0))]
errorStorage = pd.DataFrame({'MSEin': errorStorage[0], 'MSEfinal': errorStorage[1], 'MSEcomp': errorStorage[2]})
completeResult = fullstorage.mean(axis=0)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(MSEcomp)
plt.legend(modelNames)
plt.title('MSEcomp')
plt.show()

errorStorage = errorStorage / repititions

fig, ax = plt.subplots(1,1)
ax.axis('tight')
ax.axis('off')
ax.table(cellText=errorStorage.values,colLabels=errorStorage.columns,rowLabels=['OLSHAC','OLSHAC All Subsets','Lasso', 'SeqFS', 'ANNFS', 'Broad ANN'],loc="center")
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(inflationIdx.iloc[maxLag:,0].reset_index(drop=True))
plt.plot(completeResult)
plt.title(modelSelected + ' Best Results ' + inflationTicker + ' ' + str(yearOutOilChange) + '# YoY Price Change')
plt.show()

##
# Error exam
Y = np.array(inflationIdx.iloc[maxLag:,0])
# Deltas (Level)
deltaYs = pd.DataFrame(np.ones([len(Y)-1,3]))
for i in range(0,len(Y)-1):
    deltaYs.iloc[i,0] = Y[i+1] - Y[i]
    deltaYs.iloc[i,1] = completeResult[i+1] - completeResult[i]
    deltaYs.iloc[i,2] = deltaYs.iloc[i,1] - deltaYs.iloc[i,0]
    
plt.figure(figsize=[15,10])
plt.grid(True)
plt.scatter(deltaYs.iloc[:,0],deltaYs.iloc[:,1])
plt.title('X is real delta, Y is projected')
plt.plot(np.unique(deltaYs.iloc[:,0]), np.poly1d(np.polyfit(deltaYs.iloc[:,0], deltaYs.iloc[:,1], 1))(np.unique(deltaYs.iloc[:,0])))
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.scatter(deltaYs.iloc[:,0],deltaYs.iloc[:,2])
plt.title('X is real delta, Y is residual')
plt.plot(np.unique(deltaYs.iloc[:,0]), np.poly1d(np.polyfit(deltaYs.iloc[:,0], deltaYs.iloc[:,2], 1))(np.unique(deltaYs.iloc[:,0])))
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.scatter(deltaYs.iloc[:,1],deltaYs.iloc[:,2])
plt.title('X is projected delta, Y is residual')
plt.plot(np.unique(deltaYs.iloc[:,1]), np.poly1d(np.polyfit(deltaYs.iloc[:,1], deltaYs.iloc[:,2], 1))(np.unique(deltaYs.iloc[:,1])))
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(deltaYs.iloc[:,2])
plt.title('Residual through time')
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
x = np.random.normal(10, 5, size=10000)
bin_heights, bin_borders, _ = plt.hist(deltaYs.iloc[:,2], bins='auto')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), linewidth = 4)
plt.title('Residual distribution')
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
importantFeatures = FSstorage.sum(axis=0)
plt.bar([i for i in range(0,len(np.array(importantFeatures)))],np.array(importantFeatures))
plt.title('X months lag')
plt.show()
# Output
#export1 = pd.DataFrame(completeResult).set_index(importantDates)
#export1.to_excel(r'C:\Users\MI2\Desktop\Excel\USCPIBrent.xlsx', index = True)