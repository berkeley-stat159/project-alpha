"""
Creates plots from our analysis of model selection script
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import sys # instead of os
import os
import matplotlib.pyplot as plt

aic=np.loadtxt("../data/model_comparison/AIC_2.txt")
bic=np.loadtxt("../data/model_comparison/BIC_2.txt")
adjR2=np.loadtxt("../data/model_comparison/AdjR2_2.txt")



##################
# First AIC PLOT #
##################

plt.plot([1,2,3,4,4.5,5],aic[0,:],label="all conditions together")
plt.plot([1,2,3,4,4.5,5],aic[1,:],label="individual conditions")
plt.title("AIC")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig('../../images/aic.png')
plt.close()

###################
# First  BIC PLOT #
###################

plt.plot([1,2,3,4,4.5,5],bic[0,:],label="all conditions together")
plt.plot([1,2,3,4,4.5,5],bic[1,:],label="individual conditions")
plt.title("BIC")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig('../../images/bic.png')
plt.close()


##########################
# First Adjusted R2 Plot #
##########################

plt.plot([1,2,3,4,4.5,5],adjR2[0,:],label="all conditions conditions")
plt.plot([1,2,3,4,4.5,5],adjR2[1,:],label="individual conditions")
plt.title("Adjusted R2")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig('../../images/adjr2.png')
plt.close()

np.round(aic,3)
np.round(bic,3)
np.round(adjR2,3)


# making the plots make more sense:
names=["HRF","HRF+DRIFT","HRF+DRIFT \n +FOURIER6","HRF+DRIFT\n +PCA4","HRF+DRIFT\n +PCA6","HRF+DRIFT\n+FOURIER6+PCA6"]


aic_better=aic.copy()
bic_better=bic.copy()
adjR2_better=adjR2.copy()
aic_better[:,3],aic_better[:,4]     = aic[:,4],aic[:,3]
bic_better[:,3],bic_better[:,4]     = bic[:,4],bic[:,3]
adjR2_better[:,3],adjR2_better[:,4] = adjR2[:,4],adjR2[:,3]


#####################
# Improved AIC PLOT #
#####################

plt.plot(np.arange(6)+1,aic_better[0,:],label="all conditions together",linestyle='-', marker='o')
plt.plot(np.arange(6)+1,aic_better[1,:],label="individual conditions",linestyle='-', marker='o')
x=np.arange(6)+1
labels = names
plt.xticks(x, labels)
plt.title("AIC")
plt.xlabel("Model Features (all with intercept term)")
plt.ylabel("AIC metric")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig('../../images/aic_better.png',bbox_inches='tight')
plt.close()

#####################
# Improved BIC PLOT #
#####################

plt.plot(np.arange(6)+1,bic_better[0,:],label="all conditions together",linestyle='-', marker='o')
plt.plot(np.arange(6)+1,bic_better[1,:],label="individual conditions",linestyle='-', marker='o')
x=np.arange(6)+1
labels = names
plt.xticks(x, labels)
plt.xlabel("Model Features (all with intercept term)")
plt.ylabel("BIC metric")
plt.title("BIC")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig('../../images/bic_better.png',bbox_inches='tight')
plt.close()


##########################
# Improved Adjusted R2 Plot #
##########################

plt.plot(np.arange(6)+1,adjR2_better[0,:],label="all conditions together",linestyle='-', marker='o')
plt.plot(np.arange(6)+1,adjR2_better[1,:],label="individual conditions",linestyle='-', marker='o')
plt.title("Adjusted R^2")
plt.xlabel("Model Features (all with intercept term)")
plt.ylabel("Adjusted R^2 metric")
x=np.arange(6)+1
labels = names
plt.xticks(x, labels)
plt.legend(loc='upper left', shadow=True,fontsize="smaller")
plt.savefig('../../images/adjr2_better.png',bbox_inches='tight')
plt.close()


