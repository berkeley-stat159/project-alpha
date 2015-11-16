""" Tests for noise_correction functions in noise_correction.py
This checks the convolution function against the np.convolve build in function
when data follows the assumptions under np.convolve. 

Run at the project directory with:
    nosetests code/utils/tests/test_noise_correction.py
"""
# Loading modules.
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys 
import os
import scipy.stats
from scipy.stats import gamma
from numpy.testing import assert_almost_equal, assert_array_equal
from nose.tools import assert_not_equals

# Path to the subject 009 fMRI data used in class. 
location_of_data="data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_to_class_data="data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load our noise correction functions. 
from noise_correction import mean_underlying_noise,fourier_creation,fourier_predict_underlying_noise
# Load GLM functions. 
from glm import glm, glm_diagnostics, glm_multiple

def test_noise_correction():
	# tests mean_underlying_noise
	# Case where there is noise.
	test=np.arange(256)
	test=test.reshape((4,4,4,4))
	val=np.mean(np.arange(0,256,4))
	y_mean = mean_underlying_noise(test)
	assert(all(y_mean==(np.tile(val,4)+np.array([0,1,2,3]))   ))

	# Case where there is no noise.
	test_2=np.ones(256)
	test_2=test_2.reshape((4,4,4,4))
	y_mean2 = mean_underlying_noise(test_2)
	assert(all(y_mean2==np.ones(4)))
	
	# Test predicting noise with Fourier series. 
	fourier_X, fourier_MRSS, fourier_fitted, fourier_residuals = fourier_predict_underlying_noise(y_mean, 10)	
	naive_resid = y_mean-y_mean.mean()
	assert_not_equals(naive_resid[0], fourier_residuals[0])




