# displays for the convolution appendix

import numpy as np
import matplotlib.pyplot as plt
import sys


project_location= "../../../"
functions=project_location +"code/utils/functions/"

location_of_created_images=project_location+"paper/appendix/convolution/images/"

sys.path.append(functions)


from event_related_fMRI_functions import hrf_single,convolution_specialized

one_zeros = np.zeros(40)
one_zeros[4] = 1 
one_zeros[16:20]=1


plt.scatter(np.arange(40),one_zeros)
plt.xlim(-1,40)
plt.title("Stimulus pattern")
plt.savefig(location_of_created_images+"on_off_pattern.png")
plt.close()


plt.plot(np.linspace(0,30,200),np.array([hrf_single(x) for x in np.linspace(0,30,200)]))
plt.title("Single HRF, started at t=0")
plt.savefig(location_of_created_images+"hrf_pattern.png")
plt.close()

convolved=convolution_specialized(np.arange(40),one_zeros,hrf_single,np.linspace(0,60,300))
plt.plot(np.linspace(0,60,300),convolved)
plt.title("Convolution")
plt.savefig(location_of_created_images+"initial_convolved.png")
plt.close()