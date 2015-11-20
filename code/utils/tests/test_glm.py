""" Tests for glm function in glm module
This checks the glm function with the procedure in the "Basic linear 
modeling" exercise from Day 14. 

Run at the project directory with:
    nosetests code/utils/tests/test_glm.py
"""
# Loading modules.
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class. 
pathtoclassdata = "data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load our GLM functions. 
from glm import glm, glm_diagnostics, glm_multiple

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

    # Pick a single voxel to check diagnostics. 
    # Calculate actual fitted values, residuals, and MRSS of voxel.
    actual_fitted = actual_design.dot(actual_B_4d[42, 32, 19])
    actual_residuals = data[42, 32, 19] - actual_fitted
    actual_MRSS = np.sum(actual_residuals**2)/(actual_design.shape[0] - npl.matrix_rank(actual_design))
    
    # Calculate using glm_diagnostics function.
    exp_MRSS, exp_fitted, exp_residuals = glm_diagnostics(exp_B_4d, exp_design, data)
    assert_almost_equal(actual_fitted, exp_fitted[42, 32, 19])
    assert_almost_equal(actual_residuals, exp_residuals[42, 32, 19])
    assert_almost_equal(actual_MRSS, exp_MRSS[42, 32, 19])

def test_glm_multiple(): 
    # example from http://www.jarrodmillman.com/rcsds/lectures/glm_intro.html
    # it should be pointed out that hypothesis just looks at simple linear regression

    psychopathy = [11.416,   4.514,  12.204,  14.835,
    8.416,   6.563,  17.343, 13.02,
    15.19 ,  11.902,  22.721,  22.324]
    clammy = [0.389,  0.2  ,  0.241,  0.463,
    4.585,  1.097,  1.642,  4.972,
    7.957,  5.585,  5.527,  6.964]  
    berkeley_indicator = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    stanford_indicator = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    mit_indicator      = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    schools=[berkeley_indicator,stanford_indicator,mit_indicator]

    Y = np.array([psychopathy,clammy])

    X = np.ones((len(berkeley_indicator),3)) # we aren't including the [1] as a column here

    for i,school in enumerate(schools):
        X[:,i]=school

    b,X =glm_multiple(Y,X)

    # from lecture notes
    assert round(b[0,0],5) == 10.74225
    assert round(b[0,1],5) == 11.3355
    assert round(b[0,2],5) == 18.03425
    
    


