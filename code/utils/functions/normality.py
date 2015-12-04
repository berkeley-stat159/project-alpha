import numpy as np
from scipy.stats import shapiro
from scipy.stats.mstats import kruskalwallis
    
def check_sw(resid_4d): 
    """
    Shapiro-Wilk tests the null hypothesis that the data was drawn 
    from a normal distribution. In particular, this function 
    performs a Shapiro-Wilk test on each voxel's residuals. 
    Parameters
    ---------
    resid_4d: residual data of 4D numpy array
    
    Returns
    -------
    sw_normality: 3D array of p-values.
    
    """
    sw_3d = np.zeros(resid_4d.shape[:-1])
    for i in range(resid_4d.shape[0]):
        for j in range(resid_4d.shape[1]):
            for k in range(resid_4d.shape[2]):
                junk, sw_3d[i,j,k] = shapiro(resid_4d[i,j,k,:])
    return sw_3d


def check_sw_masked(resid_2d): 
    """
    Shapiro-Wilk tests the null hypothesis that the data was drawn 
    from a normal distribution. In particular, this function 
    performs a Shapiro-Wilk test on each voxel's residuals. Same as 
    check_sw, but assumes that the residuals come in a 2D array, 
    with voxels as the rows. 
    Parameters
    ---------
    resid_2d: residual data of 2D numpy array
    
    Returns
    -------
    sw_normality: 1D array of p-values.
    
    """
    sw_1d = np.zeros(resid_2d.shape[0])
    for i in range(resid_2d.shape[0]):
        junk, sw_1d[i] = shapiro(resid_2d[i,:])
    return sw_1d
            

def check_kw(resid_4d): 
    """
    Kruskal-Wallis tests the null hypothesis that the population 
    median of all of the groups are equal. In particular, this 
    function performs a Kruskal-Wallis test for each voxel's 
    residuals against a sample from the normal distribution. 

    Parameters
    ---------
    resid_4d: residual data of 4D numpy array
    
    Returns
    -------
    kw_normality: 3D array of p-values.
    
    """
    kw_3d = np.zeros(resid_4d.shape[:-1])
    for i in range(resid_4d.shape[0]):
        for j in range(resid_4d.shape[1]):
            for k in range(resid_4d.shape[2]):
                norm_samp = np.random.normal(np.mean(resid_4d[i,j,k,:]), np.std(resid_4d[i,j,k,:]), resid_4d.shape[-1])
                junk, kw_3d[i,j,k] = kruskalwallis(resid_4d[i,j,k,:], norm_samp)
    return kw_3d
 




