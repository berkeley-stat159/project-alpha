from scipy.stats import t as t_dist
from glm import glm
import numpy as np
#import numpy.linalg as npl
#from hypothesis import t_stat

# Order the p-values from smallest to largest
# p-value has a rank that is the index of itself in the ordered array
# critical value = (i/m)*Q where i = rank, m = number of tests, Q = FDR 
# Compare each p-value to its critical value 
# The largest p-value where p < (i/m)*Q is significant, and so are all the p-values before it 

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
	k = Q/len(p_vals)

	# Multiply an array of rank values by k
	upper = k*np.fromiter(range(1 + len(p_vals)), dtype = "int")

	p_sorted = np.sort(p_vals)
	p_sorted = np.ravel(p_sorted)

	bool_array = np.zeros(len(p_sorted), dtype = bool)
	for i in range(len(p_sorted)):
		if np.all(p_sorted[i] < upper[i]):
			bool_array[i] = 1

	# Find maximum True index and the element in it from p_sorted
	max_upper = p_sorted[max(max(np.where(bool_array)))]


	# Make all non-siginificant p-values zero
	final_p = [x if x <= max_upper else 1 for x in p_vals]
	return np.array(final_p)

