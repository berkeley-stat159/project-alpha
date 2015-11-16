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
peak_value = gamma.pdf(4.91, 6)
undershoot_value = gamma.pdf(4.91, 12)
max_value = peak_value - 0.35 * undershoot_value

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


	
def fast_hrf(values):
	""" Return values for HRF at multiple values

	Parameters:
	-----------
	value: an array-like structure of integers or floats.

	Returns:
	--------
	comb_values: the hrf(values) evaluated 


	Note:
	-----
	You must change the max_value (use np.argmax) if you change the function
	"""

	# Gamma pdf for the peak
	peak_values = gamma.pdf(values, 6)
	# Gamma pdf for the undershoot
	undershoot_values = gamma.pdf(values, 12)
	# Combine them
	comb_values = peak_values - 0.35 * undershoot_values
	# if outside the range of the function
	comb_values[np.logical_or(values <0, values >30)] = 0
	# Scale max to 0.6
	return comb_values / max_value * 0.6 
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

	X = np.zeros((cuts*(N-1)+1,2))
	X[:,0] = np.linspace(min(record_cuts),max(record_cuts),num=cuts*(N-1)+1)

	for i,time in enumerate(real_times): # single for-loop ( O(cuts*n)   )
		min_close=np.min(abs(X[:,0]-time))     
		X[(abs(X[:,0]-time)==min_close),1]=1*on_off[i]

	neural_X=X
	# now use np.convolve with better thing
	hrf_discrete=np.array([hrf_function(x) for x in np.arange(0, 30+TR/cuts, TR/cuts)]) # num since the hrf function takes 30 seconds to finish 
	#np.arange(0, 30, TR/cuts)
	larger_output = np.convolve(X[:,1],hrf_discrete) # too many cuts
	N2= X.shape[0]

	larger_output=larger_output[:N2]

	desired_x_i=min(record_cuts) +cuts*np.arange(len(record_cuts))
	
	output = larger_output[list(desired_x_i)]

	return output,neural_X



# faster convolution_specialized
def fast_convolution(real_times,on_off,hrf_function,record_cuts):

	""" Does convolution on Event-Related fMRI data, assumes non-constant/non-fixed time slices, takes in fMRI recorded cuts. Uses matrix multiplication, so it's fastered than convolution_specialized. 

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
		output_vector[i] = on_off.dot(hrf_function(record_cuts[i]-real_times))

	return output_vector




def create_stimuli_from_all_values(cond1,cond2,cond3):
	""" creates a sorted np.array for all stimulis in the condition files 
	
	Parameters:
	-----------
	three np.arrays (with the the times in the first column)

	Returns:
	--------
	x_s_array = a sorted np.array (1 dimensional) of all times in all condition files
	gap_between = the difference between t_i and t_{i+1}
	colors = list of color codes of the different times (corresponding to condition file number)

	"""


	x=np.hstack((cond1[:,0],cond2[:,0],cond3[:,0]))
	y=np.zeros((cond1.shape[0]+cond2.shape[0]+cond3.shape[0],))
	y[cond1.shape[0]:]=1
	y[(cond1.shape[0]+cond2.shape[0]):]+=1


	xy=zip(x,y)
	xy_sorted=sorted(xy,key= lambda x:x[0])

	x_s,y_s=zip(*xy_sorted)

	x_s_array=np.array([x for x in x_s])
	gap_between=(x_s_array[1:]-x_s_array[:-1])

	dictionary_color={0.:"red",1.:"blue",2.:"green"}
	colors=[dictionary_color[elem] for elem in y_s]

	return x_s_array, gap_between, colors







