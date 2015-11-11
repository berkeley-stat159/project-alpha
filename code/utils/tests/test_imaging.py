""" Tests for glm function in glm module
This checks the glm function with the procedure in the "Basic linear 
modeling" exercise from Day 14. 
Run with:
    nosetests test_glm.py
"""
# Loading modules.
import numpy as np
import nibabel as nib
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

# Load our visualization functions. 
from Image_Visualizing import present_3d, make_mask


def test_present():
    # Read in the image data.
    data = np.arange(100000)
    data = data.reshape((100,100,10))
    
    full=present_3d(data)
    
    assert full.shape == (400,300)
    
    
def test_mask(): 
    # example from http://www.jarrodmillman.com/rcsds/lectures/glm_intro.html
    # it should be pointed out that hypothesis just looks at simple linear regression

    data = np.arange(1000000)
    data = data.reshape((100,100,100))
    mask1 = np.ones((100,100,100))
    mask2 = np.zeros((100,100,100))
    mask3 = np.ones((200,200,100))
    
    assert_equal(make_mask(data, mask1), data) 
    assert_equal(make_mask(data,mask2), mask2)
    assert_equal(make_mask(data,mask3,fit=True).shape, data.shape)
    


    


