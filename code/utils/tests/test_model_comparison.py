""" Tests for model_comparison functions 

Run at the project directory with:
    nosetests test_model_comparion.py
"""

# Loading modules.
import numpy as np
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))


from model_comparison import adjR2, BIC, AIC, BIC_2, AIC_2

def test_1():
	# adjusted R^2 for the trivial case (RSS=TSS)
	
	y_1d = np.array([1,-1])
	df   = 1
	MRSS = np.sum((y_1d-np.mean(y_1d))**2)/df
	rank = 1


	joy  = adjR2(MRSS,y_1d,df,rank)

	assert(joy==0)

def test_2():

	y_1d = np.array([1,-1])
	df   = 1
	MRSS = np.sum((y_1d-np.mean(y_1d))**2)/df
	rank = 1
	RSS  = MRSS
	n    = 2

	joy2 = n * np.log(RSS/n) + np.log(n)*(n-df)

	assert(joy2==BIC(MRSS,y_1d,rank,df))


def test_3():

	y_1d = np.array([1,-1])
	df   = 1
	MRSS = np.sum((y_1d-np.mean(y_1d))**2)/df
	rank = 1
	RSS  = MRSS
	n    = 2

	joy3 = n * np.log(RSS/n) + 2*(n-df)

	assert(joy3==AIC(MRSS,y_1d,rank,df))


def test_4():
	# vectorization
	y_1d = np.array([1,-1,1,-2]).reshape((2,2))
	df   = 1
	MRSS = np.sum((y_1d-np.mean(y_1d))**2)/df
	rank = 1
	RSS  = MRSS
	n    = 2

	joy2 = n * np.log(RSS/n) + np.log(n)*(n-df)

	assert(joy2==BIC_2(MRSS,y_1d,rank,df))


def test_5():
	# vectorization
	y_1d = np.array([1,-1,1,-2]).reshape((2,2))
	df   = 1
	MRSS = np.sum((y_1d-np.mean(y_1d))**2)/df
	rank = 1
	RSS  = MRSS
	n    = 2

	joy3 = n * np.log(RSS/n) + 2*(n-df)

	assert(joy3==AIC_2(MRSS,y_1d,rank,df))
