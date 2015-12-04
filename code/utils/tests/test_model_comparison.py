""" Tests for model_comparison functions 

Run at the project directory with:
    nosetests test_model_comparion.py
"""

# Loading modules.
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


sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))


from model_comparison import adjR2, BIC

def test_1():
	# adjusted R^2 for the trivial case (RSS=TSS)
	
	y_1d= np.array([1,-1])
	MRSS= np.mean((y_1d-np.mean(y_1d))**2)*2
	df = 1

	joy=adjR2(MRSS,y_1d,df)

	assert(joy==0)

def test_2():

	y_1d= np.array([1,-1])
	MRSS= np.mean((y_1d-np.mean(y_1d))**2)*2
	df = 1
	RSS=MRSS
	n=2
	joy2=n * np.log(RSS/n) + np.log(n)*(n-df)

	assert(joy2==BIC(MRSS,y_1d,df))

