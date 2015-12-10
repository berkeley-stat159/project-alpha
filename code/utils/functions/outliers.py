""" 
A collection of utility functions for identifying outliers in fMRI data. Most of these are copied from janewliang's HW2 solutions. compare_outliers goes through the procedure to compare mean MRSS before and after dropping the extended outliers. 
"""
import numpy as np
from glm import glm_multiple, glm_diagnostics

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def vol_std(data):
    """ Return standard deviation across voxels for 4D array `data`

    Parameters
    ----------
    data : 4D array
        4D array from FMRI run with last axis indexing volumes.  Call the shape
        of this array (M, N, P, T) where T is the number of volumes.

    Returns
    -------
    std_values : array shape (T,)
        One dimensonal array where ``std_values[i]`` gives the standard
        deviation of all voxels contained in ``data[..., i]``.
    """
    data_2d = data.reshape((-1,data.shape[-1]))
    return data_2d.std(0)


def iqr_outliers(arr_1d, iqr_scale=1.5):
    """ Return indices of outliers identified by interquartile range

    Parameters
    ----------
    arr_1d : 1D array
        One-dimensional numpy array, from which we will identify outlier
        values.
    iqr_scale : float, optional
        Scaling for IQR to set low and high thresholds.  Low threshold is given
        by 25th centile value minus ``iqr_scale * IQR``, and high threshold id
        given by 75 centile value plus ``iqr_scale * IQR``.

    Returns
    -------
    outlier_indices : array
        Array containing indices in `arr_1d` that contain outlier values.
    lo_hi_thresh : tuple
        Tuple containing 2 values (low threshold, high thresold) as described
        above.
    """
    # Hint : np.lookfor('centile')
    # Hint : np.lookfor('nonzero')
    q25, q75 = np.percentile(arr_1d, [25 ,75])
    IQR = q75 - q25
    lo_hi_thresh = (q25-iqr_scale*IQR, q75+iqr_scale*IQR)
    outlier_indices = np.nonzero((arr_1d < lo_hi_thresh[0]) | (arr_1d > lo_hi_thresh[1]))[0]
    return outlier_indices, lo_hi_thresh


def vol_rms_diff(arr_4d):
    """ Return root mean square of differences between sequential volumes

    Parameters
    ----------
    data : 4D array
        4D array from FMRI run with last axis indexing volumes.  Call the shape
        of this array (M, N, P, T) where T is the number of volumes.

    Returns
    -------
    rms_values : array shape (T-1,)
        One dimensonal array where ``rms_values[i]`` gives the square root of
        the mean (across voxels) of the squared difference between volume i and
        volume i + 1.
    """
    diff_vols = np.diff(arr_4d.reshape((-1,arr_4d.shape[-1])))
    return np.sqrt(np.mean(diff_vols ** 2, 0))


def extend_diff_outliers(diff_indices):
    """ Extend difference-based outlier indices `diff_indices` by pairing

    Parameters
    ----------
    diff_indices : array
        Array of indices of differences that have been detected as outliers.  A
        difference index of ``i`` refers to the difference between volume ``i``
        and volume ``i + 1``.

    Returns
    -------
    extended_indices : array
        Array where each index ``j`` in `diff_indices has been replaced by two
        indices, ``j`` and ``j+1``, unless ``j+1`` is present in
        ``diff_indices``.  For example, if the input was ``[3, 7, 8, 12, 20]``,
        ``[3, 4, 7, 8, 9, 12, 13, 20, 21]``.
    """
    extended_indices = []
    for idx in diff_indices:
        if idx not in extended_indices:
            extended_indices.append(idx)
        if idx+1 not in extended_indices:
            extended_indices.append(idx+1)
    return extended_indices

def compare_outliers(data, conv, plot=False):
    """ Return standard deviation across voxels for 4D array `data`

    Parameters
    ----------
    data : 4D array
        4D array from FMRI run with last axis indexing volumes.  
    conv : 2D array of the convolved time course

    Returns
    -------
    meanMRSS : mean MRSS of simple regression on the convolved time course. 
    outmeanMRSS : mean MRSS of simple regression on the convolved time course after dropping the extended rms outliers. 
    """
    rms_values = vol_rms_diff(data)
    rms_outliers, rms_thresholds = iqr_outliers(rms_values)
    extended_indices = extend_diff_outliers(rms_outliers)

    X = np.ones((len(conv), 2))
    X[:, 1] = conv

    B, junk = glm_multiple(data, X)
    MRSS, fitted, residuals = glm_diagnostics(B, X, data)
    meanMRSS = np.mean(MRSS)

    mask = np.ones(X.shape[0])
    mask[extended_indices] = 0
    outX = X[mask.nonzero()[0],:]

    outB, junk = glm_multiple(data[...,mask.nonzero()[0]], outX)
    outMRSS, outfitted, outresiduals = glm_diagnostics(outB, outX, data[...,mask.nonzero()[0]])
    outmeanMRSS = np.mean(outMRSS)
    
    if plot==True:
        rms_values = np.resize(rms_values, len(rms_values)+1)
        rms_values[-1] = 0
        plt.plot(rms_values, "k")
        plt.plot(extended_indices, rms_values[extended_indices], "ro")
        plt.axhline(rms_thresholds[0], ls="--")
        plt.axhline(rms_thresholds[1], ls="--")

        hand_out = mlines.Line2D([], [], color="r", marker="o", ls="None", label="Outliers")
        hand_thresh = mlines.Line2D([], [], color="b", ls="--", label="Thresholds")
        plt.legend(handles=[hand_out, hand_thresh], numpoints=1)
    
    return meanMRSS, outmeanMRSS
