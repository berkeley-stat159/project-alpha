""" Tests for smoothvoxels in smooth module

Run with:
    nosetests test_smooth.py
"""

import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
from numpy.testing import assert_almost_equal
from nose.tools import assert_not_equals
# Path to the subject 009 fMRI data used in class. 
# You need to add the convolution, .nii, and condition files. 
# Assume that this is in the data directory for our project, 
# in a directory called 'ds114'. 
pathtoclassdata = "../data/ds114/"

sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load smoothing function
from smooth import smoothvoxels

def test_smooth():
	# Read in the image data.
	img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
	data = img.get_data()[..., 4:]

	# Run the smoothvoxels function with fwhm = 0 (No smoothing) at time 7
	non_smoothed_data = smoothvoxels(data, 0, 7)

	# assert that data at time 7 and non_smoothed_data are equal since fwhm = 0
	assert_almost_equal(data[..., 7], non_smoothed_data)
	
	# Run the smoothvoxels function with fwhm = 5 at time 7
	smoothed_data = smoothvoxels(data, 5, 7)
	# assert that data at time 7 and smoothed_data are not equal
	assert_not_equals(data[..., 7].all(), smoothed_data.all())
