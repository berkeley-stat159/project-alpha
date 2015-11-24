""" Tests for normality checks

Run inside the project directory with:
    nosetests code/utils/tests/test_normality.py
"""

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
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class. 
pathtoclassdata = "data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load our GLM functions. 
from glm import glm, glm_diagnostics, glm_multiple

#Load our Normality functions
from normality import sw, kw

def test_sw():
    #Reading in data image
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    
    #1D convolved time course
    TR = 2
    tr_times = np.arange(0, 30, TR)
    hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
    n_vols = data.shape[-1]

    neural_prediction = events2neural(condition_location+"cond001.txt",TR,n_vols)
    convolved = np.convolve(neural_prediction, hrf_at_trs) 
    N = len(neural_prediction)  
    M = len(hrf_at_trs) 
    np_hrf=convolved[:N]
    
    #Via GLM function
    np_B, np_X = glm(data, np_hrf)
    
    #Via GLM_Diagnostics function
    np_MRSS, np_fitted, np_residuals = glm_diagnostics(np_B, np_X, data)
    
    #run the Shapiro-Wilks function
    sw_normality = sw(np_residuals)
    assert_almost_equal(sw_normality, data[64, 64, 30, :])
    
    
def test_kw():
    #Reading in data image
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    
    #run the Kruskal-Wallis function
    sw_normality = sw(residuals)
    assert_almost_equal(sw_normality)