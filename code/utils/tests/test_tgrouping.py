""" Tests for tgrouping functions 

Run at the project directory with:
    nosetests code/utils/tests/test_tgrouping.py
"""

# Loading modules.
import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys

from numpy.testing import assert_equal

sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))


from mask_phase_2_dimension_change import neighbor_smoothing 
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end
from tgrouping import t_binary_grouping, t_grouping_neighbor


def test_1():
	x = np.arange(100)
	hope = t_binary_grouping(x, 50, prop = False, abs_on = False)

	check = np.zeros(100)
	check[50:] = 1

	np.testing.assert_equal(check, hope[0])
	x2 = 50 - x

	joy = t_binary_grouping(x2, .5, prop = True, abs_on = True)

	assert(sum(joy[0]) == 51)

def test_2():
	x = np.arange(100)
	x2 = 50 - x
	mask = np.zeros((5,5,4))
	mask[:, 1:4, 1:3] = 1
	x3 = x.reshape((5, 5, 4))
	output3 = t_grouping_neighbor(x3, mask, 50, neighbors = 1,
						prop = False, abs_on = False, binary = True , off_value = 0,masked_value = .5)

	assert(np.sum(output3[0]) == 50)


	mask[:, 1:4, 1:3] = 1
	x4 = x2.reshape((5, 5, 4))
	output4 = t_grouping_neighbor(x4, mask, .5, neighbors = 2,
						prop = True, abs_on = True, binary = True , off_value = 0, masked_value = .5)

	assert(np.sum(output4[0]) == 50)

	# Test for when neighbors != None and binary == False
	output5 = t_grouping_neighbor(x4, mask, .5, neighbors = 2,
						prop = True, abs_on = True, binary = False, off_value = 0, masked_value = .5)
	assert(output5 == False)

	# Test for when binary == False
	output6 = t_grouping_neighbor(x4, mask, .5, neighbors = None,
						prop = True, abs_on = True, binary = False, off_value = 0, masked_value = .5)
	assert(np.sum(output6[0]) == 66)
