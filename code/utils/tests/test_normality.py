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
    np.random.seed(159)
    sim_resids = np.random.rand(2, 2, 2, 100)
    sim_resids[0,0,0] = np.random.randn(100)
    
    sw_3d = check_sw(sim_resids)
    kw_3d = check_kw(sim_resids)
    print(sw_3d)
    print(kw_3d)
    
    assert(sw_3d[0,0,0] > 0.05)
    assert(sw_3d[1,0,0] < 0.05)
    
    assert(kw_3d[0,0,0] > 0.05)
    assert(kw_3d[1,0,0] < 0.05)
