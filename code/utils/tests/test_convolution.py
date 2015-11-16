""" Tests for convolution function in event_related_fMRI_function model
This checks the convolution function against the np.convolve build in function
when data follows the assumptions under np.convolve. 

Run at the project directory with:
    nosetests code/utils/tests/test_convolution.py
"""
# Loading modules.
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys 
import os
import scipy.stats
from scipy.stats import gamma
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class.
location_to_class_data="data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load our convolution functions. 
from event_related_fMRI_functions import convolution, convolution_specialized, hrf_single, np_convolve_30_cuts, fast_convolution,fast_hrf,create_stimuli_from_all_values
from stimuli import events2neural

def test_convolution():
	#################
	# i. Can the user-created functions match np.convolve in np.convolve territory

	TR = 2.5
	tr_times = np.arange(0, 30, TR)
	hrf_at_trs = np.array([hrf_single(x) for x in tr_times])

	n_vols = 173
	neural_prediction = events2neural(location_to_class_data+'ds114_sub009_t2r1_cond.txt',TR,n_vols)
	all_tr_times = np.arange(173) * TR


	##################
	# a. np.convolve #
	##################


	testconv_np = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
	N = len(neural_prediction)  # N == n_vols == 173
	M = len(hrf_at_trs)  # M == 12
	testconv_np=testconv_np[:N]

	#####################
	# b. user functions #
	#####################

	#--------#
	# second #

	testconv_2 = convolution(all_tr_times,neural_prediction,hrf_single)


	#-------#
	# third #

	testconv_3 = convolution_specialized(all_tr_times,neural_prediction,
		hrf_single,all_tr_times)


	#--------#
	# fourth #

	on_off = np.zeros(174)
	real_times,on_off[:-1] = np.linspace(0,432.5,173+1),neural_prediction
	hrf_function,TR,record_cuts= hrf_single, 2.5 ,np.linspace(0,432.5,173+1)
	#
	testconv_4_1 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=1)[0]

	testconv_4_15 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=15)[0]


	testconv_4_30 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30)[0]


	#-------#
	# fifth #

	testconv_5 = fast_convolution(all_tr_times,neural_prediction,fast_hrf,all_tr_times)

	additional_runs=[testconv_np,testconv_2,testconv_3,testconv_4_1,testconv_4_15,testconv_4_30,testconv_5]
	names=["testconv_np","testconv_2","testconv_3","testconv_4_1","testconv_4_15","testconv_4_30","testconv_5"]
	print("Max difference between model and testconv_np:")
	for i,my_convolved in enumerate(additional_runs):
		if my_convolved.shape[0]==testconv_np.shape[0]:
			print(names[i],max(abs(testconv_np-my_convolved)))
		else:
			print(names[i],max(abs(testconv_np-my_convolved[:-1])))


# Actual asserts
	for i,my_convolved in enumerate(additional_runs):
		if my_convolved.shape[0]==testconv_np.shape[0]:
			assert (max(abs(testconv_np-my_convolved) < .0001))
		else:
			assert (max(abs(testconv_np-my_convolved[:-1]) < .0001))



	
def test_convolution_specialized():
	stimuli=np.array([0,5,15])
	on_off1=np.array([0,1,0])
	on_off2=np.array([1,0,1])
	x=np.linspace(0,45,91) # 0, .5, 1, 1.5, 2, ... 45
	HRF1=convolution_specialized(stimuli,on_off1,hrf_single,x)
	y1=np.array([hrf_single(x_i-5) for x_i in x]) # what it should be doing
	HRF2=convolution_specialized(stimuli,on_off2,hrf_single,x)
	y2=np.array([hrf_single(x_i)+hrf_single(x_i-15) for x_i in x]) #what it should be doing

	assert all(HRF1 == y1)
	assert all(HRF2 == y2)


# This function is currently not in use.
def test_create_stimuli_from_all_values():
	# Simulate some conditions. 
	c1 = np.arange(1,51).reshape(10,-1)
	c2 = np.arange(21,71).reshape(10,-1)
	c3 = np.arange(11,61).reshape(10,-1)
	
	# Get the sorted array, gaps, and colors. 
	x_s_array, gap_between, colors = create_stimuli_from_all_values(c1, c2, c3)

	assert(np.diff(x_s_array).all() == gap_between.all())
