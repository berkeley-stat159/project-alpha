from mask_phase_2_dimension_change import neighbor_smoothing 
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end
import numpy as np

def t_binary_grouping(t, cutoff, prop = False, abs_on = False):
	
    """
	Evaluated the t values above a cutoff or proportion

	Parameters
	----------
    t:         t-value of the betas 1d numpy array
    cutoff:    the limit for the false discovery rate
    prop:      logical~ if the cutoff is a proportion or a value
    abs_on:    logical~ if we want to take absolute value of the t input


	Returns
    -------
    zero_one:  vector of ones and zeros where ones are above the cutoff, and zeros are below
    cutoff: the limit for the false discovery rate

    Notes
    -----
    If you want the values to be preserved multiply t*zero_one afterwards
	"""
	# if you want to use proportion you'll need to provide a logical cutoff value
	assert(0 <= cutoff*prop and cutoff*prop <= 1)

	# just to be safe:
	t= np.ravel(t)

	# if we'd like to take into account abs(t)
	if abs_on:
		t = np.abs(t)
	
	# sexy shorting
	t_sorted = np.sort(t)
	
	if prop:
		num = int((1 - cutoff)*t.shape[0])
		cutoff = t_sorted[num]

	zero_one = np.zeros(t.shape)
	zero_one[t >= cutoff] = 1
		
	return zero_one, cutoff


def t_grouping_neighbor(t_3d, mask, cutoff, neighbors = None,
						prop = False, abs_on = False, binary = True, off_value = 0, masked_value = .5):
	"""
	Masks a 3d array, does t_binary_grouping, and does neighboring 

	Parameters
	----------
    t_3d:      t-value of the betas 3d numpy array
    mask:      a 3d numpy array of 0s and 1s that has the same shape as t_3d
    cutoff:    the limit for the false discovery rate
    neighbors: number of neighbors for neighbor smoothing (must have binary be true)

    prop:      logical~ if the cutoff is a proportion or a value
    abs_on:    logical~ if we want to take absolute value of the t input
    binary:    if binary, then off_value is ignored and 0 is used as the 
    			off_value, 1 as the on value
    off_value: the value of those not selected


	Returns
    -------
    output_3d:  a 3d numpy array same size as the t_3d with either:
    			 (1) binary on_off values for inside the mask and "masked_value" 
    			 for values outside mask or (2) t values to the accepted values, 
    			 and "off_values" for lost values, and "masked_value" for values 
    			 outside mask. MOREOVER, it can have had neighbor smoothing applied 
    			 the binary case
    cutoff: the limit for the false discovery rate

	"""
	if neighbors != None and binary == False:
		return False

	t_1d = masking_reshape_start(t_3d, mask)
	t_1d = np.ravel(t_1d)
	zero_one, cutoff = t_binary_grouping(t_1d, cutoff, prop, abs_on)

	if not binary:
		t_1d = t_1d*zero_one + off_value*(1 - zero_one)
	else:
		t_1d = zero_one

	output_3d = masking_reshape_end(t_1d, mask, masked_value)

	if neighbors != None:
		output_3d = neighbor_smoothing(output_3d, neighbors)

	return output_3d, cutoff
