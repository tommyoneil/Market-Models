# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:00:52 2021

@author: MI2
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from fitnetFunction import *

def fitnetFunction(inputs, targets, dset):
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets) #same as matlab code training with inputs and targets
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    MLP = MLPRegressor(hidden_layer_sizes=(10,10,10,10,10), max_iter=2000)
    MLP.fit(X_train, y_train.values.ravel())
    
    final = scaler.transform(dset.iloc[:,1:])
    predictions = MLP.predict(final)
    
    return predictions