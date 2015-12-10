# masking_reshape_functions.py
# this file provides a way to mask data, then reduce it's dimensions 
# (and then create the correct output after analysis is done on 1d to 2d data)
import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys


def masking_reshape_start(data, mask):
	"""
	takes a 3 or 4d data and utilizes a mask to return a 1 or 2d reshaped output

	Input:
	------
	data: 3d *or* 4d np.array  (x,y,z) or (x,y,z,t) shape
	mask: a 3d np array  (x,y,z) shape, with values 0s and 1s (1 desired, 0 remove)

	Returns:
	--------
	reshaped: a 1d *or* 2d np.array (connected to 3d or 4d "data" input)

	"""
	assert(len(data.shape) == 3 or len(data.shape) == 4)

	mask_1d=np.ravel(mask)
	b_mask_1d = (mask_1d==1)


	if len(data.shape) == 3:
		data_1d = np.ravel(data)
		reshaped = data_1d[b_mask_1d]

	if len(data.shape) == 4:
		data_2d = data.reshape((-1, data.shape[-1]))
		reshaped = data_2d[b_mask_1d, :]
	return reshaped



def masking_reshape_end(data_small, mask, off_value=0):
	"""
	takes a 1d input, utilizes a mask to convert into 3d output ()

	Notes:
	------
	mask must have same number of ones as the data_small.shape[0]

	Input:
	------
	data_small: a 1d np.array
	mask:       a 3d np array  (x,y,z) shape, with values 0s and 1s (1 desired, 0 remove), see notes
	off_value:  the value to be replaced for the non-on values of the mask


	Returns:
	--------
	data_big: 3d np.array  (x,y,z) shape

	"""
	assert(len(data_small.shape) == 1)

	data_big = off_value*np.ones((mask.shape))

	data_big[mask == 1] = data_small

	return data_big

def neighbor_smoothing(data_3d, neighbors):
	"""
	takes a 3d input, returns mini-smoothing depending on positivity of neighbors
	Notes:
	------
	data_3d must be 3-dimensional
	Input:
	------
	data_3d: a 3d np.array
	neighbors: the value that indicates the number of neighbors around voxel to check
	Returns:
	--------
	smoothed_neighbors: 3d np.array  (x,y,z) shape
	"""
	smoothed_neighbors = data_3d
	off = np.max(data_3d)
	#print(off) 
	shape = data_3d.shape

	# these for loops fail the travis coverage!!!
	for i in 1 + np.arange(shape[0] - 2):
		for j in 1 + np.arange(shape[1] - 2):
			for k in 1 + np.arange(shape[2] - 2):
				# number of neighbors that need to be positivednm
				if np.sum(data_3d[(i - 1):(i + 2),(j - 1):(j + 2),(k - 1):(k + 2)] < 0) < neighbors and data_3d[i, j, k] < 0:
					smoothed_neighbors[i, j, k] = off
	return smoothed_neighbors


def neighbor_smoothing_binary(data_3d, neighbors):
	"""
	takes a 3d binary (0/1) input, returns a "neighbor" smoothed 3d matrix


	Input:
	------
	data_3d:   a 3d np.array (with 0s and 1s) -> 1s are "on", 0s are "off"
	neighbors: the value that indicates the number of neighbors around voxel to check

	Returns:
	--------
	smoothed_neighbors: 3d np.array same shape as data_3d

	"""
	smoothed_neighbors = data_3d.copy()

	shape = data_3d.shape

	for i in 1 + np.arange(shape[0] - 2):
		for j in 1 + np.arange(shape[1] - 2):
			for k in 1 + np.arange(shape[2] - 2):
				# number of neighbors that need to be positivednm
				if np.sum(data_3d[(i - 1):(i + 2),(j - 1):(j + 2),(k - 1):(k + 2)] == 1) < neighbors and data_3d[i, j, k] == 1:
					smoothed_neighbors[i, j, k] = 0

	return smoothed_neighbors



