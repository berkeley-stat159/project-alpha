import numpy as np

def time_shift(convolved, neural_prediction, delta):
    """ Returns tuple containing original convolved time course 
    with the correct number of volumes and a back-shifted 
    convolved time course. 
    
    Parameters:
    -----------
    convolved: 1-d array of the convolved time course.
    neural_prediction: 1-d array of the event stimuli
    
    delta: a single numeric value indicating how much to shift.
    
    Returns:
    --------
    convolved2: convolved time course, but only up to the number 
    of volumes
    
    shifted: convolved time course, back-shifted by delta. 
    """
    # Number of volumes.
    N = len(neural_prediction) 
    # Assert that the shifting factor is reasonable.
    assert(delta+N <= len(convolved))

    # Knock off the extra volumes. 
    convolved2 = convolved[:N]
    # Backshift by delta. 
    shifted = convolved[delta:(delta+N)]
    return convolved2, shifted


def time_shift_cond(condition, delta):
    """ Returns the shifted condition file
    
    Parameters:
    -----------
    condition: a 1d np.array of stimulus times
    
    delta: a single numeric value indicating how much to shift.
    
    Returns:
    --------
    shift_condition: 1d np.array time shifted conditional data
    """
    
    shift_condition= condition-delta
    
    return shift_condition



def make_shift_matrix(condition,delta_vector):
    """ Returns a matrix of shifted conditions as the columns (depending upon delta_vector)
    
    Parameters:
    -----------
    condition: a 1d np.array of stimulus times (length n)
    
    delta_vector: a 1d np.array of shifts (length m)
    
    Returns:
    --------
    shift_matrix: a 2d np.array with time shifted columns (n x m)
    """
    m = len(delta_vector)
    n = condition.shape[0]
    X=np.ones((n,m))

    shifts=-X*delta_vector

    shift_matrix=np.tile(condition,m).reshape(m,n).T+shifts
    return shift_matrix



def time_correct(convolve_lambda,shift_matrix,num_TRs):
    """ Returns a prediction for the Hemodyamic response for the given time points 
    
    Parameters:
    -----------
    convolution_lambda: function that takes in 1 parameter (a 1d vector of times to be convolved)
    
    shift_matrix: a 2d np.array with time shifted columns (n x m)
    num_TRs: expected dimension of convolve_lambda output
    
    Returns:
    --------
    hrf_matrix: a 2d np.array with predicted hrf
    """
    hrf_matrix=np.zeros((num_TRs,shift_matrix.shape[1]))
    for i in range(shift_matrix.shape[1]):

        hrf_matrix[:,i]=convolve_lambda(shift_matrix[:,i])

    return hrf_matrix

