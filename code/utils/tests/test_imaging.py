""" Tests for functions in imaging module

Run at the project directory with:
    nosetests code/utils/tests/test_imaging.py
"""
# Loading modules.
import numpy as np
import nibabel as nib
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load our visualization functions. 
from Image_Visualizing import present_3d, make_mask,present_3d_options

# all tests of present are looking at the output sizes of the 2d arrays
def test_present():
    # Read in the image data.
    data = np.arange(100000)
    data = data.reshape((100,100,10))
    
    full=present_3d(data)
    
    assert full.shape == (400,300)

def test_present_options_2():
    data = np.arange(100000)
    data = data.reshape((100,100,10))
    
    full=present_3d_options(data,axis=2)
    
    first=np.ceil(np.sqrt(10))
    second=np.ceil(10/first)

    assert full.shape == (100*first,100*second)

def test_present_options_1():
    data = np.arange(100000)
    data = data.reshape((100,100,10))
    
    full=present_3d_options(data,axis=1)
    assert full.shape == (10*10,100*10)

def test_present_options_0():
    data = np.arange(100000)
    data = data.reshape((100,100,10))
    
    full=present_3d_options(data,axis=0)
    assert full.shape == (10*10,100*10)

    
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
    
    x= False
    try:
        make_mask(data,mask3,fit=False)
    except ValueError:
        x=True
    assert(x==True)
    


    


