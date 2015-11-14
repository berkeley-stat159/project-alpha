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
from numpy.testing import assert_almost_equal, assert_array_equal
# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append("../functions")

# Load smoothing function
from smooth import smoothvoxels

def test_smooth():
	# Read in the image data.
	img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
	data = img.get_data()[..., 4:]

	# arbitrary 4d array of ones
	#ones_array = np.ones((3, 3, 3, 3))

	# Run the smoothvoxels function with fwhm = 0 (No smoothing) at time 7
	non_smoothed_data = smoothvoxels(data, 0, 7)

	# assert that data at time 7 and non_smoothed_data are equal since fwhm = 0
	assert_almost_equal(data[..., 7], non_smoothed_data)