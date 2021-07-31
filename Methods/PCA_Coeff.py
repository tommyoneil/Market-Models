# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:57:12 2021

@author: MI2
"""
import numpy as np
import linalg
from linalg import *  # import the Matrix class and utility functions top-level
from linalg import Matrix  # import the Matrix class
from numpy import linalg as LA
import pandas as pd

def PCA_Coeff(DF):
    """ performs principal components analysis 
    (PCA) on the n-by-p data matrix A
    Rows of A correspond to observations, columns to variables. 

    Returns :  
    coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
    score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (DF-np.mean(DF.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = LA.eig(M.cov()) # attention:not always sorted
    return coeff