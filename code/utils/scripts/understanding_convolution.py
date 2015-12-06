################################################################
# Welcome to "using the correct convolution function" bootcamp #
################################################################

# First off, this function isn't the only function that is really necessary,
# it interacts with the time shift functions, and an example of that can be
# see in the final script as well as the beginning time shift script.

#######################################################
# 'Ello there, here's the function your' looking for: #

# np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30)



##################################
# with the following doc-string: #
""" Does convolution on Event-Related fMRI data, cutting TR into 'cuts' equal 
distance chunks and putting stimulus in closed cut 

Parameters:
-----------
real_times   = one dimensional np.array of time slices (size K)
on_off       = a one dimensional np.array of the on_off values (not necessarily 
    1s and 0s)
hrf_function = a hrf (in functional form, not as a vector)
TR           = time between record_cuts
record_cuts  = vector with fMRI times that it recorded (size N)
cuts         = number of cuts between each TR

Returns:
--------
output_vector = vector of hrf predicted values (size N)

Note:
-----
It should be noted that you can make the output vector (size "N+M+1") if you'd 
like by adding on extra elements in the times and have their on_off values be 0 
at the end of both
"""


# So the above was written by a stupid person, so let's be clear in our context:

# 1. the "real_times" will the cond_all or cond_i one dimensional np.array 
# (so you'll have to change what you read in if you load the file in, taking 
#    just the first column)

# 2. the "on_off" will just be a np.ones(len(real_times)) for us, that is: a
# 1d array of all ones, length of "real_times"

# 3. The hrf_function should be "hrf_single", which you can get from the same
# function file (event_related_fMRI_functions.py)

# 4. TR =2 (for us at the very list)

# 5. record_cuts: is really just np.arange(n_vols)*2    (where n_vols in the 
#    number of time slices)


# 6. cuts: in our case, it will actually be 15.

#######
# Examples of this function can be found in the script file (both initial
#    and the final convolution script)
#######







