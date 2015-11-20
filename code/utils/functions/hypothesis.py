#hypothesis testing function
from scipy.stats import t as t_dist
from glm import glm, glm_multiple
import numpy as np
import numpy.linalg as npl

def t_stat(data_4d, convolved, c = [0,1]):
    """
    Return four values, the estimated beta, t-value, 
    degrees of freedom, and p-value for the given t-value
    
    Parameters
    ----------
    data_4d: numpy array of 4 dimensions 
        The image data of one subject
    convolved: numpy array of 1 dimension
        The convolved time course
    c: numpy array of 1 dimension
        The contrast vector fo the weights of the beta vector. 
        Default is [0,1] which corresponds to beta_1

    Note that the fourth dimension of `data_4d` (time or the number 
    of volumes) must be the same as the length of `convolved`. 
    
    Returns
    -------
    beta: estimated beta values
    
    t: numpy array of 1 dimension
        t-value of the betas
    
    df: int
        degrees of freedom
    
    p: numpy array of 1 dimension
        p-value corresponding to the t-value and degrees of freedom
    """

    # Make sure y, X, c are all arrays
    beta, X = glm(data_4d, convolved)
    c = np.atleast_2d(c).T  # As column vector
    # Calculate the parameters - b hat
    beta = np.reshape(beta, (-1, beta.shape[-1])).T

    fitted = X.dot(beta)
    # Residual error
    y = np.reshape(data_4d, (-1, data_4d.shape[-1]))
    errors = y.T - fitted
    # Residual sum of squares
    RSS = (errors**2).sum(axis=0)
 
    df = X.shape[0] - npl.matrix_rank(X)
    # Mean residual sum of squares
    MRSS = RSS / df
    # calculate bottom half of t statistic
    
    SE = np.sqrt(MRSS * c.T.dot(npl.pinv(X.T.dot(X)).dot(c)))
    zeros = np.where(SE==0)
    SE[zeros] = 1
    t = c.T.dot(beta) / SE

    t[:,zeros] =0
    # Get p value for t value using cumulative density dunction
    # (CDF) of t distribution
    ltp = t_dist.cdf(abs(t), df) # lower tail p
    p = 1 - ltp # upper tail p
    
    return beta, t, df, p
        


def t_stat_mult_regression_single(data_4d, X, c = () ):
    """
    Return four values, the estimated beta, t-value, 
    degrees of freedom, and p-value for the given t-value
    
    Parameters
    ----------
    data_4d: numpy array of 4 dimensions 
        The image data of one subject
    X: numpy array 
        the matrix to be put into the glm_mutiple function
    c: numpy array of 1 dimension
        The contrast vector fo the weights of the beta vector. 
        If not entered, it will be set as np.array([0,1,...]) which corresponds 
        to beta_1

    Note that the fourth dimension of `data_4d` (time or the number 
    of volumes) must be the same as the number of rows that X has. 
    
    Returns
    -------
    beta: estimated beta values
    
    t: numpy array of 1 dimension (spe)
        t-value of the betas
    
    df: int
        degrees of freedom
    
    p: numpy array of 1 dimension
        p-value corresponding to the t-value and degrees of freedom
    """

    # Make sure y, X, c are all arrays
    beta, X = glm_multiple(data_4d, X)

    # dealing with no c put in
    if c is ():
        c = np.zeros(X.shape[-1])
        c[1]=1


    c = np.atleast_2d(c).T  # As column vector


    # Calculate the parameters - b hat
    beta = np.reshape(beta, (-1, beta.shape[-1])).T

    fitted = X.dot(beta)
    # Residual error
    y = np.reshape(data_4d, (-1, data_4d.shape[-1]))
    errors = y.T - fitted
    # Residual sum of squares
    RSS = (errors**2).sum(axis=0)
 
    df = X.shape[0] - npl.matrix_rank(X)
    # Mean residual sum of squares
    MRSS = RSS / df
    # calculate bottom half of t statistic
    
    SE = np.sqrt(MRSS * c.T.dot(npl.pinv(X.T.dot(X)).dot(c)))
    zeros = np.where(SE==0)
    SE[zeros] = 1
    t = c.T.dot(beta) / SE

    t[:,zeros] =0
    # Get p value for t value using cumulative density dunction
    # (CDF) of t distribution
    ltp = t_dist.cdf(abs(t), df) # lower tail p
    p = 1 - ltp # upper tail p
    
    return beta, t, df, p
        

def t_stat_mult_regression(data_4d, X):
    """
    Return four values, the estimated beta, t-value, 
    degrees of freedom, and p-value for the given t-value
    
    Parameters
    ----------
    data_4d: numpy array of 4 dimensions 
        The image data of one subject
    X: numpy array 
        the matrix to be put into the glm_mutiple function

    Note that the fourth dimension of `data_4d` (time or the number 
    of volumes) must be the same as the number of rows that X has. 
    
    Returns
    -------
    beta: estimated beta values
    
    t: numpy array of 2 dimensions
        t-value of the betas
    
    df: int
        degrees of freedom
    
    p: numpy array of 2 dimensions
        p-value corresponding to the t-value and degrees of freedom
    """

    beta, X = glm_multiple(data_4d, X)

    # Calculate the parameters - b hat
    beta = np.reshape(beta, (-1, beta.shape[-1])).T

    fitted = X.dot(beta)
    # Residual error
    y = np.reshape(data_4d, (-1, data_4d.shape[-1]))
    errors = y.T - fitted
    # Residual sum of squares
    RSS = (errors**2).sum(axis=0)
 
    df = X.shape[0] - npl.matrix_rank(X)
    # Mean residual sum of squares
    MRSS = RSS / df
    # calculate bottom half of t statistic
    Cov_beta=npl.pinv(X.T.dot(X))

    SE =np.zeros(beta.shape)
    for i in range(X.shape[-1]):
        c = np.zeros(X.shape[-1])
        c[i]=1
        c = np.atleast_2d(c).T
        SE[i,:]= np.sqrt(MRSS* c.T.dot(npl.pinv(X.T.dot(X)).dot(c)))


    zeros = np.where(SE==0)
    SE[zeros] = 1
    t = beta / SE

    t[:,zeros] =0
    # Get p value for t value using cumulative density dunction
    # (CDF) of t distribution
    ltp = t_dist.cdf(abs(t), df) # lower tail p
    p = 1 - ltp # upper tail p
    
    return beta.T, t, df, p



