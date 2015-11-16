""" Tests for hypothesis testing function in hypothesis_test module
This checks the hypothesis testing which closely follows the "General
Linear Models Lecture from class"

Run with:
    nosetests test_hypothesis.py
"""
# Loading modules.
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import sys, os
from numpy.testing import assert_almost_equal, assert_array_equal

# Path to the subject 009 fMRI data used in class.  
# Path to functions. 
pathtoclassdata = "../data/ds114/"
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

# Load our t_stat functions. 
from hypothesis import t_stat,t_stat_mult_regression_single,t_stat_mult_regression


def test_hypothesis1():
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    # Read in the convolutions. 
    convolved = np.loadtxt(pathtoclassdata + "ds114_sub009_t2r1_conv.txt")[4:]
    # Create design matrix. 
    
    beta,t,df,p = t_stat(data, convolved,[1,1])
    beta2, t2,df2,p2 = t_stat(data, convolved,[0,1])
    
    assert_almost_equal(beta,beta2)
    assert t.all() == t2.all()
    assert beta.shape[1] == np.prod(data.shape[0:-1])


def test_hypothesis2():
    # example from http://www.jarrodmillman.com/rcsds/lectures/glm_intro.html
    # it should be pointed out that hypothesis just looks at simple linear 
    # regression

    psychopathy = [11.416,   4.514,  12.204,  14.835,
    8.416,   6.563,  17.343, 13.02,
    15.19 ,  11.902,  22.721,  22.324]
    clammy = [0.389,  0.2  ,  0.241,  0.463,
    4.585,  1.097,  1.642,  4.972,
    7.957,  5.585,  5.527,  6.964]  

    Y = np.asarray(psychopathy)
    B, t, df, p = t_stat(Y, clammy, [0, 1])


    assert np.round(t,6)==np.array([[ 1.914389]])
    assert np.round(p,6)==np.array([[ 0.042295]])


def test_hypothesis_3():
    # new multiple-regression
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    # Read in the convolutions. 
    convolved = np.loadtxt(pathtoclassdata + "ds114_sub009_t2r1_conv.txt")[4:]
    # Create design matrix. 
    X=np.ones((convolved.shape[0],2))
    X[:,1]=convolved


    beta,t,df,p = t_stat(data, convolved,[0,1])
    beta2, t2,df2,p2 = t_stat_mult_regression_single(data, X,np.array([0,1]))

    beta3, t3,df3,p3 = t_stat_mult_regression(data, X)


    assert_array_equal(t,t2)
    assert_array_equal(t,np.atleast_2d(t3[1,:]))
    

