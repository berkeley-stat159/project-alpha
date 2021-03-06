from __future__ import absolute_import, division, print_function
import numpy as np


def bh_procedure(p_vals, Q):
	"""
	Return an array (mask) of the significant, valid tests
		out of the p-values. not significant p-values are denoted by ones.

	Parameters
	----------
    p_vals: p-values from the t_stat function (1-dimensional array)

    Q: The false discovery rate 


	Returns
    -------
    significant_pvals : 1-d array of p-values of tests that are
    	deemed significant, denoted by 1's and p-values

    Note: You will have to reshape the output to the shape of the data set.
	"""
	# k is Q/m where m = len(p_vals)
	k = Q/p_vals.shape[0]

	# Multiply an array of rank values by k
	upper = k*np.fromiter(range(1, 1 + p_vals.shape[0]), dtype = "int")

	p_sorted = np.sort(p_vals, axis = 0)

	bool_array = np.zeros(p_sorted.shape[0], dtype = bool)
	for i in range(p_sorted.shape[0]):
		if p_sorted[i] < upper[i]:
			bool_array[i] = True

	# Find maximum True index and the element in it from p_sorted
	indices = np.where(bool_array)

	# Make sure there are indices that returned True!!
	if sum(indices[0]) != 0:
		max_true_index = np.max(indices)
		# max_upper is the highest that a p-value can be to be considered significant.
		max_upper = np.ravel(p_sorted)[max_true_index]
	# If no indices where p < upper 
    
	else:
		print("**** Oh no. No p-values smaller than upper bound FDR were found. ****")
		return p_vals

	# Make all non-siginificant p-values zero
	final_p = [x if x <= max_upper else 1 for x in np.ravel(p_vals)]
	# shape of returned array is (len,)
	return np.array(final_p)

