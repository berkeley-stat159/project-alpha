## Scripts to Perform Intermediate Analysis

This repository stores the script files to generate all intermediate analysis. 

- `bh_script.py`: Runs Benjamini-Hochberg on a single subject's "p-values" 
from a linear model. Compares the results with and without masking and 
different levels of neighbor smoothing. 
- `cluster.py`: Attempts agglomerative clustering on a single subject's 
t-statistics. 
- `convolution_appendix_plots.py`: Generates plots for the convolution 
appendix in the paper. 
- `event_related_HRF_script.py`: Compares the behavior of several different 
convolution approaches. 
- `get_pcs_script.py`: Obtains the principal components for the voxel by time 
matrix. 
- `glm_script.py`: Runs linear regression for a single subject, using some 
early attempts at convolution. 
- `hypothesis_script.py`: Performs t-tests on the resulting coefficients from 
running linear regression on a single subject. 
- `lin_reg_plots.py`: Generates plots relating to linear regression for the 
paper. 
- `mean_across.py`: Averages t-statistics per voxel from linear regression 
across subjects. 
- `model_selection.py`: Compares several different linear regression models 
using AIC, BIC, and adjusted R-squared. 
- `multi_regression_script.py`: Some earlier comparisons of different linear 
regression models with different convolution approaches and inclusion of 
conditions for a single subject. 
- `noise_correction_script.py`: Experiments with noise correction techniques 
for a single subject. 
- `normality_script.py`: Performs the Shapiro-Wilk test for normality on the 
residuals from running linear regression on a single subject. 
- `outliers_script.py`: Identifies potential outliers for each subject using 
the HW2 technique. 
- `pca_script.py`: Examines proportion of variance explained by principal 
components of the voxel by time matrix for each subject. 
- `smooth_script.py`: Spatially smooths a single subject's voxels. 
- `tgrouping_script.py`: Evaluate t-statistics corresponding to a single 
subject's linear regression coefficients. 
- `time_shift_script.py`: Examines the effects of implementing a time 
correction in the event-related voxel responses for a single subject. 
- `tsa_script.py`: Fits a single voxel time course to an ARIMA process. 

Additionally, `cluster.npy` and `cluster_mask.npy` are binary files storing 
the unmasked and masked clustering data referenced in `cluster.py`.

