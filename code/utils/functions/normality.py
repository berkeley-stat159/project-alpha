import numpy as np
from scipy.stats import gamma
from scipy.stats import mstats
from functools import wraps
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
import scipy.stats
import os
import sys


#Wrapping of a function with decorators
def wrap(pre, post):
    def decorate(func):
        def call(*args, **kwargs):
            pre(func, *args, **kwargs)
            result = func(*args, **kwargs)
            post(func, *args, **kwargs)
            return result
        return call
    return decorate 

def trace_in(func, *args, **kwargs):
    print "Entering normality test",  func.__name__
    
def trace_out(func, *args, **kwargs):
    print "Leaving normality test", func.__name__
    
#Wrapping Effect
@wrap(trace_in, trace_out)
def sw(data_4d): #Shapiro Wilks
    """
    Parameters
    ---------
    data_4d: residual data of 4D numpy array
    
    Returns
    -------
    sw_normality: test statistic from Shapiro-Wilks normality test
    
    """
    if i in range(64):
        if j in range(64):
            if k in range(34):
                sw_normality = scipy.stats.shapiro(data_4d[i,j,k ,:])
                return sw_normality
            

def kw(data_4d): #Kruskal-Wallis
    """
    Parameters
    ---------
    data_4d: residual data of 4D numpy array
    
    Returns
    -------
    kw_normality: test statistic from Kruskal-Wallis normality test
    
    """
    if i in range(64):
        if j in range(64):
            if k in range(34):
                kw_normality = scipy.stats.mstats.kruskalwallis(data_4d[i,j,k,:])
                return kw_normality
 




