""" Tests the time_shift function.
Run with:
    nosetests test_time_shift.py
"""

# Loading modules.
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys 
from numpy.testing import assert_almost_equal
import nibabel as nib

# Path to the subject 009 fMRI data used in class. 
pathtoclassdata = "data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

from stimuli import events2neural
from event_related_fMRI_functions import hrf_single
from time_shift import time_shift,time_shift_cond,make_shift_matrix, time_correct


def test_time_shift():
    # Intialize values for class data. 
    TR = 2.5
    tr_times = np.arange(0, 30, TR)
    hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
    
    # Load class data.
    n_vols = 173
    neural_prediction = events2neural(pathtoclassdata+'ds114_sub009_t2r1_cond.txt', TR, n_vols)

    # Get np.convolve time course. 
    convolved = np.convolve(neural_prediction, hrf_at_trs) 
    N = len(neural_prediction)  # N == n_vols == 173

    # Compare shifted time courses by hand and using function.
    actual_shifted = convolved[5:(5+N)]
    exp_convolved2, exp_shifted = time_shift(convolved, neural_prediction, 5)
    assert_almost_equal(actual_shifted, exp_shifted)


def times_time_shift_2():
    # time_shift_cond
    assert(np.all(np.arange(5)-1==time_shift_cond(np.arange(5),1) ))

    # make_shift_matrix
    assert(np.all(make_shift_matrix(np.arange(5),np.arange(2))==
        np.array([0,1,2,3,4, 
                 -1,0,1,2,3]).reshape((2,-1)).T))

    
