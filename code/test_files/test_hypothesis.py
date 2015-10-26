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
import sys, os
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class.  
# Path to functions. 
pathtoclassdata = "../../data/ds114/"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

# Load our t_stat functions. 
from hypothesis import t_stat

def test_hypothesis():
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    # Read in the convolutions. 
    convolved = np.loadtxt(pathtoclassdata + "ds114_sub009_t2r1_conv.txt")[4:]
    # Create design matrix. 
    
    beta,t,df,p = t_stat(data, convolved,[1,1])
    beta2, t2,df2,p2 = t_stat(data, convolved,[0,1])
    
    assert_almost_equal(beta,beta2)
    assert t.all() == t2.all()
    assert beta.shape[1] == np.prod(data.shape[0:-1]) 
    
    

    
    
    
    