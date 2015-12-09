
# coding: utf-8

# # BART trial

# This is some initial exploration and analysis related to the Bart Trials. I hope the comments make sense :).  Since this is ipython I've intermixed *bash* code with the *python* code, I hope this is easy to follow.

# In[1]:

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import os # new
# the last one is a major thing for ipython notebook, don't include in regular python code
get_ipython().magic('matplotlib inline')


# Quickly (some rational for additions):
#  - pandas: is good because it has Data frame structures similar to R data.frames (I've already make some CSV files using this library)
#  - os: is good for file location (instead of trying to use $\texttt{bash}$ into $\texttt{ipython}$)
#    - $\texttt{os.chdir}$ $\Leftrightarrow$ $\texttt{cd}$ in $\texttt{bash}$
#    - $\texttt{os.listdir}$ $\Leftrightarrow$ $\texttt{ls}$   -> usually I do $\texttt{np.array(os.listdir(...))}$ if the directory is large
# 

# ## Layout of File System, also exploring $\texttt{os}$ library

# Below I've provided where my data files are currently located.  You may observe that the numbering of the subjects is missing some numbers.

# In[6]:

# setting locations of elements, make sure to change this: (smart idea in general, when dealing with a file system_
location_of_data="/Users/BenjaminLeRoy/Desktop/1.Fall2015/Stat 159/project/data/ds009/" 
location_of_subject001=os.path.join(location_of_data,"sub001/")
location_of_simuli="/Users/BenjaminLeRoy/Desktop/test/4d_fmri/"
location_of_present_3d="/Users/BenjaminLeRoy/Desktop/1.Fall2015/Stat 159/project/python_code"
location_of_processed_data="/Users/BenjaminLeRoy/Desktop/1.Fall2015/Stat 159/project/processed_data/"


# ### Folders in the large Data Directory (ds009)

# In[7]:

os.chdir(location_of_data)
np.array(os.listdir(location_of_data))
# some subject numbers don't exist (probably removed due to errors mentioned in paper)


# * Ignore the '.DS_Store', we can take it out if we ever want to deal with this as a list (see "Creating Data Frames")

# ### Examining 1 subject's data

# These is the folders in the sub001 folder:

# In[8]:

os.chdir(location_of_subject001)
np.array(os.listdir(location_of_subject001))


# ### File Structure for Individual Subject

# I've tried to help you visualize the data by using the tree function (try in your own terminal), you need your directory to be "ds009/sub001"

# In[9]:

#Run is in terminal after entering sub001 folder:
#  1)
#  tree BOLD/task001_run001
#  2)
#  tree anatomy/
#  3)
#  tree behav/task001_run001
#  4)
#  tree model/model001/onsets/task001_run001
#  5)
#  tree model/model002/onsets/task001_run001

# 1)
BOLD/task001_run001
├── QA
│   ├── DVARS.png
│   ├── QA_report.pdf
│   ├── confound.txt
│   ├── dvars.txt
│   ├── fd.png
│   ├── fd.txt
│   ├── mad.png
│   ├── maskmean.png
│   ├── meanpsd.png
│   ├── qadata.csv
│   ├── spike.png
│   ├── voxcv.png
│   ├── voxmean.png
│   └── voxsfnr.png
├── bold.nii
└── bold.nii.gz

# 2)
anatomy
├── highres001.nii.gz
├── highres001_brain.nii.gz
├── highres001_brain_mask.nii.gz
├── inplane001.nii.gz
├── inplane001_brain.nii.gz
└── inplane001_brain_mask.nii.gz

# 3)
behav/task001_run001
└── behavdata.txt

# 4)
model/model001/onsets/task001_run001
├── cond001.txt
├── cond002.txt
└── cond003.txt

# 5)
model/model002/onsets/task001_run001
├── cond001.txt
├── cond002.txt
└── cond003.txt
# Commentary on the above structure:
# 1. Explore for yourself some of the following:
#  1. that the model 001 and 002 files seem to be the same (iono why that might be)
#  2. BOLD directory ("BOLD/QA") also contains a lot of their images they produced for this individual (maybe try to reproduce?)

# ### Exploring the behavdata.txt

# I've included already created files to combine all behavioral data into CSV files, and we can load these csv files with panda.
# Below is visual of the data:

# In[10]:

os.chdir(location_of_processed_data)
behav=pd.read_table("task001_run001_model_data_frame.csv",sep=",")
behav.head(5)


# For BART, we just a have a few items in the Behavior data, and they all make a good amount of sense. Feel free to see the dictionary if you can't guess at them now (or read the pdf files).
# 
# I will make comments about $\texttt{NumExpl}$ and $\texttt{NumTRs}$ later so try to figure these out at least :)

# ## Loading in All the Libraries (and additional programs)

# I've also imported
# - events2neural which was done in class
# - present_3d a code I have already created, see the example later

# In[11]:

#location of my stimuli.py file
os.chdir(location_of_simuli)
from stimuli import events2neural


# In[12]:

#locating my Image_Visualizing.py file
os.chdir(location_of_present_3d)
from Image_Visualizing import present_3d


# ## Now for the actual loading of Files and a little Analysis

# There are some other observations below, that might be interesting to find

# In[13]:

os.chdir(os.path.join(location_of_subject001,"BOLD/task001_run001"))
img=nib.load("bold.nii")
data=img.get_data()
# data.shape # (64, 64, 34, 245)


# In[14]:

# just a single 3d image to show case the present_3d function
three_d_image=data[...,0]


# In[ ]:

# use of my specialized function
full=present_3d(three_d_image)
plt.imshow(full,cmap="gray",interpolation="nearest")
plt.colorbar()


# ## Data Exploration

# 1) Is there a major problem in the beginning of the data?   
#   *we will come comment on this later

# In[ ]:

# cut from middle of the brain
test=data[32,32,15,:] # random voxel
plt.plot(test) # doesn't look like there are problems in the morning


# 2) Looking at the Conditions/ different types of events in scans

# In[ ]:

# model (condition data) (will be used to create on/off switches)
os.chdir(os.path.join(location_of_subject001,"model/model001/onsets/task001_run001"))
cond1=np.loadtxt("cond001.txt")
cond2=np.loadtxt("cond002.txt")
cond3=np.loadtxt("cond003.txt")


# In[ ]:

# Looking at the first to understand values
cond1[:10,:]


# If you remember, there are 3 different types of conditions for the BART trial: (regular, before pop, before save)
#  - We already know how many times the first person popped the balloon (see above) *8*. So,... I'd bet money that we could figure out which is that one, and the regular should probably be the largest one. In the first draft of this I included some more analysis, but this is a pretty straight forward reason, so lets use it.
#  - In the rest of my analysis not included here we saw different lengths of time between elements- and the paper says so as well, this is slightly annoying, but we can deal with it, because we have the start values.

# In[ ]:

for i in [cond1,cond2,cond3]:
    print(i.shape)


# We should notice that the $\texttt{NumTRs}$ in the behavior file (239) is different than the time dimension of the data (245).
# 
# I've talked to Jarrod and he thinks the folks just cut out the first 6 recordings, which makes sense as a general practice, I didn't see any note of it anywhere, but Jarrod suggest looking for sumplimentary documents from the paper.

# In[ ]:

print(str(len(data[0,0,0,:]))+ " is not equal to " + str(behav["NumTRs"][0])) # not the same


# 3) Looking at conditional data in different fashions

# Problem with dimensions of fMRI data and numTRs

# In[ ]:

events=events2neural("cond001.txt",2,239) # 1s are non special events
events=np.abs(events-1) # switching 0,1 to 1,0


# In[ ]:

data_cut=data[...,6:]
# data_cut.shape (64, 64, 34, 239)


# We should approach the rest that we can do, the the dimensions are the same :) 

# ### Visualizing when the 3 conditions happen:
# and that using only the event data will seperate condition 1 from condition 2 and 3

# In[ ]:

x=np.hstack((cond1[:,0],cond2[:,0],cond3[:,0]))

# specifying which condition they come from (0,1,2)
y=np.zeros((cond1.shape[0]+cond2.shape[0]+cond3.shape[0],))
y[cond1.shape[0]:]=1
y[(cond1.shape[0]+cond2.shape[0]):]+=1

xy=zip(x,y)
xy_sorted=sorted(xy,key= lambda pair:pair[0]) #sorted by x values
x_s,y_s=zip(*xy_sorted) # unzipping
x_s_array=np.array([x for x in x_s])
desired=(x_s_array[1:]-x_s_array[:-1]) 
# difference between the element before and itself (time delay)


# setting up color coding for 3 different types of condition
dictionary_color={0.:"red",1.:"blue",2.:"green"}
colors=[dictionary_color[elem] for elem in y_s[:-1]]

#plot
plt.scatter(x_s_array[:-1]/2,desired,color=colors,label="starts of stimuli")
plt.plot(events*10,label="event neural stimili function")
#plt.plot(events*4) if it's hard to see with just the 10 function
plt.xlabel("Time, by TR")
plt.ylabel("length of time to the next value")
plt.xlim(0-5,239+5)
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.title("Just Checking")


# In[ ]:



