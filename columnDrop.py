# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:49:54 2021

@author: MI2
"""
#METHOD TO DROP NANS COLUMN BY COLUMN
import pandas as pd

def columnDrop(DF):
    NewDF = pd.DataFrame()
    for i in range(0,len(DF.columns)):
        NewDF[str(i)] = DF.iloc[:,i].dropna().reset_index(drop=True)
    return NewDF