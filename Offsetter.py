# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:22:18 2021

@author: MI2
"""
import numpy as np
import pandas as pd

def offsetter(rawData, columnRotations):
    Yin = rawData.iloc[1+int(max(columnRotations)):,0]
    X = rawData.iloc[:,1:]
    for i in range(0,len(columnRotations)):
        X.iloc[:,i] = np.roll(X.iloc[:,i], int(columnRotations[i]) - int(max(columnRotations)))
    Xin = X.iloc[0:len(Yin),:]
    XinCons = pd.DataFrame(data=Xin).reset_index(drop=True)
    XinCons['1'] = np.ones([len(Xin.iloc[:,0]),1])
    Xout = X.iloc[0:len(Yin)+int(min(columnRotations)),:]
    XoutCons = pd.DataFrame(data=Xout).reset_index(drop=True)
    XoutCons['1'] = np.ones([len(Xout.iloc[:,0]),1])
    return [Yin,Xin,XinCons,Xout,XoutCons]
