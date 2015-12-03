""" Tests for bh_procedure in benjamini_hochberg module

Run at the project directory with:
    nosetests code/utils/tests/test_mask_phase_2.py
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

#from numpy.testing import assert_almost_equal
#from nose.tools import assert_not_equals
from nose.tools import assert_equals
#from nose.tools import assert_array_equals

sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))


from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end



def test_1():
	# just checks masking_reshape_start

	# 3d
	happy=np.arange(27)
	happy=happy.reshape((3,3,3))

	mask=np.zeros((3,3,3))
	mask[:2,1:,2:]=1

	joy=masking_reshape_start(happy,mask)

	assert(np.all(joy == np.ravel(happy[:2,1:,2:])))

	# 4d
	happy4=np.arange(27*4)
	happy4=happy4.reshape((3,3,3,4))

	mask4=np.zeros((3,3,3,4))
	mask4[:2,1:,2:,:]=1

	joy4=masking_reshape_start(happy4,mask4)

	assert(np.all(np.ravel(joy4) == np.ravel(happy4[:2,1:,2:,:])))


def test_2():
	# checks BOTH masking_reshape_start and masking_reshape_end

	# 3d
	happy=np.arange(27)
	happy=happy.reshape((3,3,3))

	mask=np.zeros((3,3,3))
	mask[:2,1:,2:]=1

	joy=masking_reshape_start(happy,mask)

	output=masking_reshape_end(joy,mask,off_value=0)

	test=np.zeros((3,3,3))
	test[:2,1:,2:]=happy[:2,1:,2:]

	assert(np.all(test==output))

