# voxel underlying noise function
import numpy as np
from glm import glm, glm_diagnostics, glm_multiple

# mean_underlying_noise

def mean_underlying_noise(data_4d):
	""" takes average of data_4d across the 4th dimension (time)

	Parameters:
	-----------
	data_4d: 4 dimensional np.array 
		(with 4th dimension the one trying to take mean over)

	Returns:
	--------
	y_mean: average of data_4d across the 4th dimension
	"""
	data_2d=data_4d.reshape(np.prod(data_4d.shape[:-1]),-1)

	y_mean=np.mean(data_2d,axis=0)

	return y_mean

def fourier_creation(n,p):
	""" predicts the underlying noise using fourier series and glm

	Parameters:
	-----------
	n: desired length to run over (assumes 0:(n-1) by integers)
	p: number of fourier series

	Returns:
	--------
	X: glm_matrix (first column is all 1s) (dim p+1)

	Note:
	-----
	Does a backwards approach to fouriers (possibly sacrificing orthogonality), wants to
	look at maximum period first
	"""	

	X = np.ones((n,p+1))
	for i in range(p):
		X[:,i+1]=np.sin(((i+1)/X.shape[0])*2*np.arange(n))

	return X

def fourier_predict_underlying_noise(y_mean,p):
	""" predicts the underlying noise using fourier series and glm

	Parameters:
	-----------
	y_mean: 1 dimensional np.array
	p: number of fourier series

	Returns:
	--------
	X: glm_matrix (first column is all 1s)
	fitted: the fitted values from glm
	residuals: the residuals betwen fitted and y_mean
	MRSS: MRSS from glm function (general output from glm_diagnostics)

	Note:
	-----
	Does a backwards approach to fouriers (possibly sacrificing orthogonality), wants to
	look at maximum period first
	"""
	n= y_mean.shape[0]
	X=fourier_creation(n,p)
	beta, junk=glm_multiple(y_mean,X)
	MRSS, fitted, residuals = glm_diagnostics(beta, X, y_mean)

	return X,MRSS,fitted,residuals




