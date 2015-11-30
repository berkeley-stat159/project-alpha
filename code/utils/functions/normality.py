import numpy as np
from scipy.stats import shapiro
from scipy.stats.mstats import kruskalwallis
    
def check_sw(resid_4d): #Shapiro Wilks
    """
    Parameters
    ---------
    resid_4d: residual data of 4D numpy array
    
    Returns
    -------
    sw_normality: test statistic from Shapiro-Wilks normality test
    
    """
    sw_3d = np.zeros(resid_4d.shape[:-1])
    for i in range(resid_4d.shape[0]):
        for j in range(resid_4d.shape[1]):
            for k in range(resid_4d.shape[2]):
                junk, sw_3d[i,j,k] = shapiro(resid_4d[i,j,k,:])
    return sw_3d
            

def check_kw(resid_4d): #Kruskal-Wallis
    """
    Parameters
    ---------
    resid_4d: residual data of 4D numpy array
    
    Returns
    -------
    kw_normality: test statistic from Kruskal-Wallis normality test
    
    """
    kw_3d = np.zeros(resid_4d.shape[:-1])
    for i in range(resid_4d.shape[0]):
        for j in range(resid_4d.shape[1]):
            for k in range(resid_4d.shape[2]):
                norm_samp = np.random.normal(np.mean(resid_4d[i,j,k,:]), np.std(resid_4d[i,j,k,:]), resid_4d.shape[-1])
                junk, kw_3d[i,j,k] = kruskalwallis(resid_4d[i,j,k,:], norm_samp)
    return kw_3d
 




