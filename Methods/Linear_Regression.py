# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:13:38 2021

@author: MI2
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def market_beta(X,Y,N):
    """ 
    X = The independent variable which is the Market
    Y = The dependent variable which is the Stock
    N = The length of the Window
    
    It returns the alphas and the betas of
    the rolling regression
    """
    
    # all the observations
    obs = len(X)
    
    # initiate the betas with null values
    betas = np.full(obs, np.nan)
    
    # initiate the alphas with null values
    alphas = np.full(obs, np.nan)
    
    
    for i in range((obs-N)):
        regressor = LinearRegression()
        regressor.fit(X.to_numpy()[i : i + N+1].reshape(-1,1), Y.to_numpy()[i : i + N+1])
        
        betas[i+N]  = regressor.coef_[0]
        alphas[i+N]  = regressor.intercept_
    return(alphas, betas)