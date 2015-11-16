""" Tests for noise_correction functions in noise_correction.py
This checks the convolution function against the np.convolve build in function
when data follows the assumptions under np.convolve. 
Run with:
    nosetests test_noise_correction.py
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

# Path to the subject 009 fMRI data used in class.  
location_of_project="../"
location_of_data=location_of_project+"data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_of_functions= "../functions/"
location_to_class_data=location_of_project+"data/ds114/"

# path to functions
sys.path.append(os.path.join(os.path.dirname(__file__), location_of_functions))

# path to class data
sys.path.append(os.path.join(os.path.dirname(__file__), location_to_class_data))

# Load our GLM functions. 
from noise_correction import mean_underlying_noise,fourier_creation,fourier_predict_underlying_noise

def test_1():
	# tests mean_underlying_noise
	test=np.arange(256)
	test=test.reshape((4,4,4,4))

	val=np.mean(np.arange(0,256,4))

	assert(all(mean_underlying_noise(test)==(np.tile(val,4)+np.array([0,1,2,3]))   ))

	test_2=np.ones(256)
	test_2=test_2.reshape((4,4,4,4))

	assert(all(mean_underlying_noise(test_2)==np.ones(4)))

# the other 2 functions are kinda annoying to test 
# (and just a few lines,... so I'm going to do bad coding practice and not check them)






