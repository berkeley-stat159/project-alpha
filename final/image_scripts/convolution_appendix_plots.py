"""
Plot producing scripts for convolution appendix
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys


project_location= "../../"
functions=project_location +"code/utils/functions/"

location_of_created_images=project_location+"images/"

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








colors=["#CCCCFF","#C4C3D0","#92A1CF","#2A52BE","#003399","#120A8F","#000080","#002366"]






xx=np.linspace(0,30,3001)

i=3
one_zeros_2 = np.zeros(3001)
one_zeros_2[(2*i*100-15):(2*i*100+15)]=.6
plt.plot(xx,.6-one_zeros_2,color="black")
plt.title(" 'g' Moving Function")
plt.ylim(-.1,1)
plt.savefig(location_of_created_images+"play.png")
plt.close()


xx=np.linspace(0,30,3001)
y1 = np.array([hrf_single(x) for x in np.linspace(0,30,3001)])
plt.plot(xx,y1)


for i in range(len(colors)):
	one_zeros_2 = np.zeros(3001)
	one_zeros_2[(2*i*100-15):(2*i*100+15)]=.6
	y2 = .6-one_zeros_2
	# plt.plot(xx,y1)
	plt.plot(xx,one_zeros_2,color="black")
	plt.plot(xx,y2,color="white")
	plt.fill_between(xx,y2,y1 , facecolor=colors[i],where= y2<.6)

plt.plot([15,19.75],[.4,.4],color="red")
plt.plot([19,20],[.41,.4],color="red")
plt.plot([19,20],[.39,.4],color="red")
plt.plot([19,19.75],[.41,.4],color="red")
plt.plot([19,19.75],[.39,.4],color="red")


plt.title("Math Convolution")
plt.savefig(location_of_created_images+"math_convolved.png")
plt.close()






"""
xx=np.linspace(0,30,301)
one_zeros_2 = np.zeros(301)
one_zeros_2[58:62]=.6
y2 = .6-one_zeros_2
y1 = np.array([hrf_single(x) for x in np.linspace(0,30,301)])
plt.plot(xx,y1)
plt.plot(xx,y2,color="white")
plt.fill_between(xx,y2,y1 , facecolor="blue",where= y2<.6)
"""


