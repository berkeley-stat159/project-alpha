# Multiple Model Comparision
import numpy as np


def adjR2(MRSS,y_1d,df,rank):
	"""
	Computes a single Adjusted R^2 value for a model (high is good) 

	Input:
	------
	MRSS : Mean Squared Error
	y_1d : the y vector as a 1d np array ( n x 1)
	df   : the degrees of the model (n-p-1 generally where = is the number of 
		features)
	rank : the rank of the X feature matrix used to create the MRSS 
		(assumed to be p+1 generally, where p is the number of features)
 
	Output:
	-------
	adjR2: the adjusted R^2 value

	Comments:
	---------
	Adjusted R^2 is a comparision tool that penalizes the number of features

	"""

	n=y_1d.shape[0]
	RSS= MRSS*df
	TSS= np.sum((y_1d-np.mean(y_1d))**2)
	adjR2 = 1- ((RSS/TSS)  * ((n-1)/(n-rank))  )

	return adjR2

def AIC(MRSS,y_1d,df,rank):
	"""
	Computes a single BIC value for a model (low is good) 

	Input:
	------
	MRSS : Mean Squared Error
	y_1d : the y vector as a 1d np array ( n x 1)
	df   : the degrees of the model (n-p-1 generally where = is the number of 
	features)
	rank : the rank of the X feature matrix used to create the MRSS 
		(assumed to be p+1 generally, where p is the number of features)

	Output:
	-------
	AIC: the adjusted AIC value

	"""
	n=y_1d.shape[0]
	RSS= MRSS*df

	AIC= n * np.log(RSS/n) + 2*(rank)

	return AIC


def BIC(MRSS,y_1d,df,rank):
	"""
	Computes a single BIC value for a model (low is good) 

	Input:
	------
	MRSS : Mean Squared Error
	y_1d : the y vector as a 1d np array ( n x 1)
	df   : the degrees of the model (n-p-1 generally where = is the number of 
	features)
	rank : the rank of the X feature matrix used to create the MRSS 
		(assumed to be p+1 generally, where p is the number of features)

	Output:
	-------
	BIC: the adjusted BIC value


	Comments:
	---------
	BIC is a bayesian approach to model comparision that more strongly 
	penalizes the number of features than AIC (which was not done, but Ben
	wants a bigger penalty than Adjusted R^2 since he hates features)
	"""
	n=y_1d.shape[0]
	RSS= MRSS*df

	BIC= n * np.log(RSS/n) + np.log(n)*(rank)

	return BIC


##### Second attempt (mult-dimensional)
def AIC_2(MRSS_vec,y_2d,df,rank):
	"""
	Computes a single BIC value for a model (low is good) 

	Input:
	------
	MRSS_vec : Mean Squared Error Vector (1d np array)
	y_1d : the y vector as a 1d np array ( n x 1)
	df   : the degrees of the model (n-p-1 generally where = is the number of 
	features)
	rank : the rank of the X feature matrix used to create the MRSS 
		(assumed to be p+1 generally, where p is the number of features)

	Output:
	-------
	AIC: the adjusted AIC value vector

	"""
	n=y_2d.shape[1]
	RSS= MRSS_vec*df

	AIC= n * np.log(RSS/n) + 2*(rank)

	return AIC


def BIC_2(MRSS_vec,y_2d,df,rank):
	"""
	Computes a single BIC value for a model (low is good) 

	Input:
	------
	MRSS_vec : Mean Squared Error Vector (1d np array n)
	y_2d : the y vector as a 2d np array ( n x t)
	df   : the degrees of the model (n-p-1 generally where = is the number of 
	features)
	rank : the rank of the X feature matrix used to create the MRSS 
		(assumed to be p+1 generally, where p is the number of features)

	Output:
	-------
	BIC: the adjusted BIC value vector


	Comments:
	---------
	BIC is a bayesian approach to model comparision that more strongly 
	penalizes the number of features than AIC (which was not done, but Ben
	wants a bigger penalty than Adjusted R^2 since he hates features)
	"""
	n=y_2d.shape[1]
	RSS= MRSS_vec*df

	BIC= n * np.log(RSS/n) + np.log(n)*(rank)

	return BIC

