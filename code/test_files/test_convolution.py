""" Tests for convolution function in event_related_fMRI_function model
This checks the convolution function against the np.convolve build in function
when data follows the assumptions under np.convolve. 
Run with:
    nosetests test_convolution.py
"""
# Loading modules.
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import scipy.stats
from scipy.stats import gamma
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class.  
location_of_project="/Users/BenjaminLeRoy/Desktop/project-alpha/"
location_of_data=location_of_project+"data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_of_functions= location_of_project + "code/functions/"
location_to_class_data="/Users/BenjaminLeRoy/Desktop/test/4d_fmri/"

# path to functions
sys.path.append(location_of_functions) 

# path to class data
sys.path.append(location_to_class_data)

# Load our GLM functions. 
from event_related_fMRI_functions import convolution, hrf_single



#convolution(times,on_off,hrf_single)


def test_convolution():
	from stimuli import events2neural
	
	TR = 2.5
	tr_times = np.arange(0, 30, TR)
	hrf_at_trs = np.array([hrf_single(x) for x in tr_times])

	n_vols = 173
	neural_prediction = events2neural(location_to_class_data+'ds114_sub009_t2r1_cond.txt',TR,n_vols)
	all_tr_times = np.arange(173) * TR

	convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
	N = len(neural_prediction)  # N == n_vols == 173
	M = len(hrf_at_trs)  # M == 12
	convolved=convolved[:N]


	my_convolved=convolution(np.linspace(0,432.5,173),neural_prediction,hrf_single)

	print("you'll have to look at the plots yourself, they're pretty close")
	plt.plot(np.linspace(0,432.5,173),convolved,label="np.convolved")
	plt.plot(np.linspace(0,432.5,173),my_convolved,label="my convolution function")

	#In [5]: max(abs(convolved-my_convolved)) < .01
	#Out[5]: 0.0087853818693934826

	assert (max(abs(convolved-my_convolved)) < .01)



