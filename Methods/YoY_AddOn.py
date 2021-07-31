# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:22:58 2021

@author: MI2
"""
import pandas as pd
import numpy as np

def YoY_AddOn(dataPeriod,DF,switches):
    switches = np.array(switches)
    results = DF.copy()
    [M,N] = results.shape
    results = results.iloc[12:,:]
    for i in range(0,N):
        if switches[i] == 1:
            if all(DF.iloc[:,i] > 0):
                for j in range(0,M-12):
                    results.iloc[j,i] = (DF.iloc[j+12,i]-DF.iloc[j,i])/DF.iloc[j,i]
        elif switches[i] == 2:
            for j in range(0,M-12): 
                results.iloc[j,i] = DF.iloc[j+12,i] - DF.iloc[j,i]
                
    #nanDF = pd.DataFrame(np.nan([12,N]))
    
    for i in range(0,N):
        nanDF = []
        for j in range(0,12):
            nanDF.append(np.nan)
        final = pd.concat([pd.DataFrame(nanDF), results], sort=False)
        final = final[DF.columns]
    
    return final