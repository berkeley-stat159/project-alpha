# Function for Event-Related fMRI
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import scipy.stats
from scipy.stats import gamma
from stimuli import events2neural


# getting the peak (for this specific hrf function)
peak_values = gamma.pdf(4.91, 6)
undershoot_values = gamma.pdf(4.91, 12)
max_value = peak_values - 0.35 * undershoot_values

# fixed maximum value
def hrf_single(value):
	""" Return values for HRF at single value

	Parameters:
	-----------
	value: a single float or integer value

	Returns:
	--------
	hrf_value: the hrf(value) evaluated 


	Note:
	-----
	You must change the max_value (use np.argmax) if you change the function
	"""

	if value <0 or value >30: # if outside the range of the function
		return 0

	# Gamma pdf for the peak
	peak_values = gamma.pdf(value, 6)
	# Gamma pdf for the undershoot
	undershoot_values = gamma.pdf(value, 12)
	# Combine them
	values = peak_values - 0.35 * undershoot_values
	# Scale max to 0.6
	return values / max_value * 0.6 
	##### you must change the max_value (use np.argmax) if you change the function




def convolution(times,on_off,hrf_function):
	""" Does convolution on Event-Related fMRI data, assumes non-constant/non-fixed time slices 

	Parameters:
	-----------
	times = one dimensional np.array of time slices (size N)
	on_off =  one dimensional np.array of on/off switch or applify the hrf from the i_th 
		position (size N)
	hrf_function = a hrf (in functional form, not as a vector)

	Returns:
	--------
	output_vector = vector of hrf predicted values (size N)

	Note:
	-----
	It should be noted that you can make the output vector (size "N-M+1") if you'd like by 
	adding on extra elements in the times and have their on_off values be 0 at the end of both

	"""

	output_vector=np.zeros(len(times))
	for i in range(len(times)):
		output_vector[i]= sum([on_off[j]*hrf_function(times[i]-times[j]) for j in range(len(times))])

	return output_vector




# second take into account the desired number of cuts

def convolution_specialized(real_times,on_off,hrf_function,record_cuts):
	""" Does convolution on Event-Related fMRI data, assumes non-constant/non-fixed time slices, takes in fMRI recorded cuts 

	Parameters:
	-----------
	real_times = one dimensional np.array of time slices (size K)
	on_off =  one dimensional np.array of on/off switch or applify the hrf from the i_th real_time
		position (size K)
	hrf_function = a hrf (in functional form, not as a vector)
	record_cuts = vector with fMRI times that it recorded (size N)

	Returns:
	--------
	output_vector = vector of hrf predicted values (size N)

	Note:
	-----
	It should be noted that you can make the output vector (size "N-M+1") if you'd like by 
	adding on extra elements in the times and have their on_off values be 0 at the end of both


	"""

	output_vector=np.zeros(len(record_cuts))
	for i in range(len(record_cuts)):
		output_vector[i]= sum([on_off[j]*hrf_function(record_cuts[i]-real_times[j]) for j in range(len(real_times))])

	return output_vector



# Third attempt at convolution
def np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30):
	""" Does convolution on Event-Related fMRI data, cutting TR into 'cuts' equal distance chunks and putting stimulus in closed cut 

	Parameters:
	-----------
	real_times = one dimensional np.array of time slices (size K)
	hrf_function = a hrf (in functional form, not as a vector)
	TR = time between record_cuts
	record_cuts = vector with fMRI times that it recorded (size N)

	Returns:
	--------
	output_vector = vector of hrf predicted values (size N)

	Note:
	-----
	It should be noted that you can make the output vector (size "N-M+1") if you'd like by 
	adding on extra elements in the times and have their on_off values be 0 at the end of both

	# np.convolve(neural_prediction, hrf_at_trs)
	"""
	
	# creating 1 and 0s like in stimuli (more fine grained)
	N= len(record_cuts)

	X = np.zeros((cuts*N,2))
	X[:,0] = np.linspace(min(record_cuts),max(record_cuts),num=cuts*N)

	for i,time in enumerate(real_times):
		min_close=np.min(abs(X[:,0]-time))     
		X[(abs(X[:,0]-time)==min_close),1]=1*on_off[i]

	# now use np.convolve with better thing
	hrf_discrete=np.array([hrf_function(x) for x in np.linspace(0,30,num=cuts*30)]) # num since the hrf function takes 30 seconds to finish 
	
	larger_output = np.convolve(X[:,1],hrf_discrete) # too many cuts
	N2= X[:,1].shape[0]

	larger_output=larger_output[:N2]



	desired_x_i= [min(record_cuts)+(cuts)*i for i in range(len(record_cuts))]
	output = larger_output[desired_x_i]

	return output




