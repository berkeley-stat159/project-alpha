""" Tests for normality checks

Run inside the project directory with:
    nosetests code/utils/tests/test_normality.py
"""

import numpy as np
from scipy.stats import shapiro
from scipy.stats.mstats import kruskalwallis
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class. 
pathtoclassdata = "data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

#Load our Normality functions
from normality import check_sw, check_kw

def test_normality():
    # Generate some 4-d random uniform data. 
    # The first 3 dimensions are like voxels, the last like time. 
    np.random.seed(159)
    sim_resids = np.random.rand(2, 2, 2, 200)
    # Force one of the time courses to be standard normal. 
    sim_resids[0,0,0] = np.random.randn(200)
    
    # Do Shaprio-Wilk and Kruskal-Wallis
    sw_3d = check_sw(sim_resids)
    kw_3d = check_kw(sim_resids)
    
    assert(sw_3d[0,0,0] > 0.05)
    assert(sw_3d[1,0,0] < 0.05)
    
    assert(kw_3d[0,0,0] > 0.05)
