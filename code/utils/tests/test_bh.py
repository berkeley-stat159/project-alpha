""" Tests for bh_procedure in benjamini_hochberg module

Run at the project directory with:
    nosetests code/utils/tests/test_bh.py
"""

# Loading modules.
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
pathtoclassdata = "data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load our benjamini-hochberg function
from benjamini_hochberg import bh_procedure
from hypothesis import t_stat

def test_bh():
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    # Read in the convolutions. 
    convolved = np.loadtxt(pathtoclassdata + "ds114_sub009_t2r1_conv.txt")[4:]
    # Create design matrix. 

    beta,t,df,p = t_stat(data, convolved,[1,1])
    beta2, t2,df2,p2 = t_stat(data, convolved,[0,1])

    Q = 1.0
    pval = p.T
    useless_bh = bh_procedure(pval, Q)

    # Since the FDR is 100%, the bh_procedure should return the exact same thing as the original data.
    #assert_almost_equal(data[...,7], useless_bh[...,7])
    assert_almost_equal(np.ravel(pval), useless_bh)

    Q_real = .25
    real_bh = bh_procedure(pval, Q_real)
    #assert_not_equals(data[...,7], real_bh[...,7])
    assert(not (np.all(np.ravel(pval) != real_bh)))