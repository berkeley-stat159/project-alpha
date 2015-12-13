## User-Defined Functions Used for Analysis 

This repository stores the code files all of the user-defined functions used 
for analysis. The tests for the functions can be found in 
`project-alpha/code/utils/tests`. To run the tests, please navigate to the 
`project-alpha/code` directory and call `make test`. 

Below are brief summaries of the contents of the code files. Additional 
details can be found in their comments and docstrings. 

- `benjamini_hochberg.py`: Function to perform a Benjamini-Hochberg 
correction, given an array of "p-values" and a false discovery rate. 
- `event_related_fMRI_functions.py`: Functions related to convolving the 
event-related response for each voxel. 
- `glm.py`: Functions to build single and multiple general linear regression 
models for four-dimensional arrays of voxel responses. 
- `hypothesis.py`: Functions to perform t-tests for significance on the 
coefficients of a linear model. 
- `Image_Visualizing.py`: Functions to obtain masked data and convert 
three-dimensional data into two-dimensional slices better appropriate for 
visualizations. 
- `mask_phase_2_dimension_change.py`: Functions to obtain masked data and 
perform neighbor smoothing. 
- `model_comparison.py`: Functions to compute the adjusted R-squared, AIC, 
and BIC for a given model. 
- `noise_correction.py`: Functions to noise-correct the event-related response 
for each voxel and obtain their Fourier series. 
- `normality.py`: Functions to test for normality using Shapiro-Wilk's method, 
as well as Kruskal-Wallis (not in use). 
- `outliers.py`: Functions used to detect outliers in the fMRI data. 
Essentially a duplicate of HW2.
- `smooth.py`: Function to spatially smooth the voxel time courses. 
- `stimuli.py`: Function to obtain the predicted neural time course from an 
event file. Provided in class. 
- `tgrouping.py`: Functions to evaluate t-statistics above and below a 
provided threshold. 
- `time_shift.py`: Functions to time-correct the event-related voxel 
responses. 

