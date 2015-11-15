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
