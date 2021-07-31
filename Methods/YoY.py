# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:30:39 2021

@author: MI2
"""
import pandas as pd
import numpy as np

def YoY_Calc(dataPeriod, DF):
    if len(DF.columns) > 1:
        YDF = pd.DataFrame(np.nan, index=DF.index, columns=DF.columns)
        if dataPeriod == 'DAILY':
            for i in range(0,len(DF.columns)):
                 YDF.iloc[:,i] = DF.iloc[:,i].pct_change(252) * 100
            return YDF
        if dataPeriod == 'WEEKLY':
            for i in range(0,len(DF.columns)):
                 YDF.iloc[:,i] = DF.iloc[:,i].pct_change(52) * 100
            return YDF
        if dataPeriod == 'MONTHLY':
            for i in range(0,len(DF.columns)):
                 YDF.iloc[:,i] = DF.iloc[:,i].pct_change(12) * 100
            return YDF
        if dataPeriod == 'QUARTERLY':
            for i in range(0,len(DF.columns)):
                 YDF.iloc[:,i] = DF.iloc[:,i].pct_change(4) * 100
            return YDF
    else:
        if dataPeriod == 'DAILY':
            YoY = (DF.pct_change(252)) * 100
            return YoY
        if dataPeriod == 'WEEKLY':
            YoY = (DF.pct_change(52)) * 100
            return YoY
        if dataPeriod == 'MONTHLY':
            YoY = (DF.pct_change(12)) * 100
            return YoY
        if dataPeriod == 'QUARTERLY':
            YoY = (DF.pct_change(4)) * 100
            return YoY
    