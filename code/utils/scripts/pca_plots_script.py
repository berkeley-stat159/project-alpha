"""
Script to get additional summary statistics and comparison plots for masked PCA.
You'll need to have the output text file from running "pca_script.py" saved in 
your present working directory.

Run with: 
    python pca_plots.py

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Relative paths to project and data. 
project_path          = "../../../"
location_of_images    = project_path+"images/"

# Read in the proportions of explained variance for the maske ddata.
masked_var = pd.read_csv('masked_var.txt', sep=' ').sort_index(1)
# Cumulative sums of variances. 
cumsums = masked_var.cumsum(0)


#######################
# Plots of Components #
#######################
plt.plot(np.arange(1,11), cumsums.median(1)[:10], 'r-o')
plt.grid()
plt.axhline(y=0.4, color='k', linestyle="--")
plt.xlabel("Principal Components")
plt.title("Sum of Proportions of Variance Explained by Components")
plt.savefig(location_of_images+'pcaALL.png')
plt.close()


##########################
# Boxplots of components #
##########################
plt.boxplot(np.array(cumsums[:10]).T)
plt.scatter(np.ones((24,10))*np.arange(1,11), np.array(cumsums[:10]).T)
plt.grid()
plt.axhline(y=0.4, color='k', linestyle="--")
plt.xlabel("Principal Components")
plt.title("Sum of Proportions of Variance Explained by Components")
plt.savefig(location_of_images+'pcaBOX.png')
