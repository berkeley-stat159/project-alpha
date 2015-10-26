# Function for Event-Related fMRI
#BEN SUCKS
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import scipy.stats
from scipy.stats import gamma


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



