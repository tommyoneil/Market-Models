# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:10:48 2021

@author: MI2
"""
import numpy as np
import pandas as pd

def seriesTransform(tsData,transformationSwitches):
#seriesTransform takes ts data and converts specific columns to delta(log)
#   tsData is time series data in chronological order that needs to be
#   converted from level data to difference in log level data (may add in
#   various periods rather than just previous data)
#   transformationSwitches is a vector of 1 and 0 that tells whether a
#   column in tsData should be converted (1) or left as is (0)
#   The returned Matrix has the same dimensions as the original
#   input matrix with the desired columns modified (NaN in the first row)
    [M,N] = tsData.shape
    if len(transformationSwitches) != N:
        print('Check Switches Dimensions')

    for i in range(0,N):
        if transformationSwitches[i] == 1:
            if all(tsData.iloc[:,i]) > 0:
               temp = list(np.diff(np.log(tsData.iloc[:,i])))
               temp.insert(0,np.nan) 
               tsData.iloc[:,i] = temp
    return tsData