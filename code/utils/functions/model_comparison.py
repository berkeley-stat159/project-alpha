# Multiple Model Comparision
import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys


def adjR2(MRSS,y_1d,df,rank):
	"""
	Computes a single Adjusted R^2 value for a model (high is good) 

	Input:
	------
	MRSS : Mean Squared Error
	y_1d : the y vector as a 1d np array ( n x 1)
	df   : the degrees of the model (n-p-1 generally where = is the number of 
		features)
	rank : the rank of the X feature matrix used to create the MRSS 
		(assumed to be p+1 generally, where p is the number of features)
 
	Output:
	-------
	adjR2: the adjusted R^2 value

	Comments:
	---------
	Adjusted R^2 is a comparision tool that penalizes the number of features

	"""

	n=y_1d.shape[0]
	RSS= MRSS*df
	TSS= np.sum((y_1d-np.mean(y_1d))**2)
	adjR2 = 1- ((RSS/TSS)  * ((n-1)/(n-rank))  )

	return adjR2

def BIC(MRSS,y_1d,df,rank):
	"""
	Computes a single BIC value for a model (low is good) 

	Input:
	------
	MRSS : Mean Squared Error
	y_1d : the y vector as a 1d np array ( n x 1)
	df   : the degrees of the model (n-p-1 generally where = is the number of 
	features)
	rank : the rank of the X feature matrix used to create the MRSS 
		(assumed to be p+1 generally, where p is the number of features)

	Output:
	-------
	BIC: the adjusted BIC value


	Comments:
	---------
	BIC is a bayesian approach to model comparision that more strongly 
	penalizes the number of features than AIC (which was not done, but Ben
	wants a bigger penalty than Adjusted R^2 since he hates features)
	"""
	n=y_1d.shape[0]
	RSS= MRSS*df

	BIC= n * np.log(RSS/n) + np.log(n)*(rank)

	return BIC
