# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:22:26 2021

@author: MI2
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #normalization

def normal_cal(A):
    if np.ndim(A)!=0:
        newDF = ((A - A.min())/(A.max()-A.min()))
        return newDF
    else:
        scaler = MinMaxScaler()
        A.iloc[:,:] = scaler.fit_transform(A)
        return A
'''
    M = len(A.index)
    N = A.shape()[1]
    norma = pd.DataFrame(np.zeros([M,N]), index=A.index)
    for i in range(0,N):
        for j in range(0,M):
            norma.iloc[j,i] = (A.iloc[j,i] - min(A.iloc[:,i]))/(max(A.iloc[:,i]))-min(A.iloc[:,i])
    return norma
'''
