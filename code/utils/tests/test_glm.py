""" Tests for glm function in glm module
This checks the glm function with the procedure in the "Basic linear 
modeling" exercise from Day 14. 
Run with:
    nosetests test_glm.py
"""
# Loading modules.
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class. 
# You need to add the convolution, .nii, and condition files. 
# Assume that this is in the data directory for our project, 
# in a directory called 'ds114'. 
pathtoclassdata = "data/ds114/"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load our GLM functions. 
from glm import glm


def test_glm():
    # Read in the image data.
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    # Read in the convolutions. 
    convolved = np.loadtxt(pathtoclassdata + "ds114_sub009_t2r1_conv.txt")[4:]
    # Create design matrix. 
    actual_design = np.ones((len(convolved), 2))
    actual_design[:, 1] = convolved
    
    # Calculate betas, copied from the exercise. 
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    actual_B = npl.pinv(actual_design).dot(data_2d.T)
    actual_B_4d = np.reshape(actual_B.T, img.shape[:-1] + (-1,))
    
    # Run function.
    exp_B_4d, exp_design = glm(data, convolved)
    assert_almost_equal(actual_B_4d, exp_B_4d)
    assert_almost_equal(actual_design, exp_design)

