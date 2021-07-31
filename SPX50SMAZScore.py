# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:14:28 2021

@author: MI2
"""
import pandas as pd
from matplotlib import pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from scipy import stats
import math

dataPeriod = "DAILY"
startDate = dt(2015,1,1)
today = dt.today()

mgr = dm.BbgDataManager()

ticker = ['SPX Index'] #only works with a single security

sids = mgr[ticker]
df1 = sids.get_historical(['HT102'],startDate,today,dataPeriod)
df1.columns = df1.columns.droplevel(0)
df1 = df1.fillna(method = 'ffill')
aD = pd.concat([df1['HT102'], (100-df1['HT102'])], axis=1)
adr = [math.log(aD.iloc[i,0]/aD.iloc[i,1]) for i in range(0, len(aD.index))]
ZADR = stats.zscore(adr[-251:-1])
ZA = stats.zscore(df1['HT102'][-251:-1])
print('Z-Score of Advancing is')
print(ZA[-1])
print('                          ')
print('Z-Score of ADR:')
print(ZADR[-1])
fig, axs = plt.subplots(2)
fig.suptitle('Top: Z-Score of Advancing; Bottom: Z-score of ADR')
axs[0].plot(ZA)
axs[1].plot(ZADR)

'''
clc;
clear all;
tic
sData = history(c, 'SPX Index', 'HT102','01/01/2015',today(),'DAILY');
[aD] = [sData(:,2) 100-sData(:,2)];
[adr] = [log(aD(:,1)./aD(:,2))];
ZADR = zscore(adr(end-250:end,1));
ZA = zscore(sData(end-250:end,2));
disp('Z-Score of Advancing is');
disp(ZA(end));
disp('Z-Score of ADR is');
disp(ZADR(end));
toc
'''