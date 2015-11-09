"""
This is a script that does some time series analysis on a single voxel 
for subject 1. It relies heavily on the statsmodels module, and since 
there aren't really any built-in functions at this time, there is no 
corresponding file in the functions or tests directories. If writing 
additional functions becomes necessary, I will implement and test them 
as needed. 
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import nibabel as nib

# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
# Path to directory to save images. 
location_of_images="../../../images/"

# Load in the image for Subject 1. 
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

# Pull out a single voxel. 
voxel = data[41, 47, 2]
plt.plot(voxel) 
plt.close()
# Sort of a curve = nonconstant mean. 
# Variance also seems to be funky toward the ends. 
plt.hist(voxel)
plt.close()
# Long right tail.
qqplot(voxel, line='q')
plt.close()
# More-or-less normal, with deviations at tails.

# Box-Cox method to find best power transformation.
bc = stats.boxcox(voxel)
bc[1] # Lambda pretty close to 0, so try log transformation.  
print("Log transforming data.")

# Log transform the data. 
lvoxel = np.log(voxel)
plt.plot(lvoxel)
plt.close()
plt.hist(lvoxel)
plt.close()
qqplot(lvoxel, line='q')
plt.close()
# Plots look pretty similar, but skewness has been eliminated. 

# Try looking at the first difference. 
diff1 = lvoxel[:-1]-lvoxel[1:]
plt.plot(diff1)
plt.close()
# Mean looks like it could be constant. 
plt.hist(diff1)
plt.close()
qqplot(diff1, line='q')
plt.close()
# QQplot still shows some deviations from normality at tails. 
print("Using first difference to gain approximate stationarity.")

# Assume that the first difference is approximately normal.
# Autocorrelation plot. First lag is significant. 
sm.graphics.tsa.plot_acf(diff1, lags=20)
plt.close()
# Partial autocorrelation plot. Dies down slowly. 
sm.graphics.tsa.plot_pacf(diff1, lags=20)
plt.close()
# Might be an IMA(1, 1)
# Or, since autocorrelation also doesn't quite die out, could be an
# ARIMA model with p>0 and q>0.

# Let's look at different ARMA models. 
res = sm.tsa.arma_order_select_ic(diff1, ic=['aic', 'bic'])
res
# Both AIC and BIC suggest ARIMA(1,1,1).

# Fit an ARIMA(1,1,1).
arima111 = sm.tsa.ARIMA(lvoxel, (1,1,1)).fit()
arima111.params

# Fitted values look reasonable compared to first difference.
plt.plot(diff1)
plt.plot(arima111.resid)
plt.close()

# Residuals look normally distributed.
qqplot(arima111.resid, line='q')
plt.close()
# Autocorrelation and partial autocorrelation plots look fine. 
sm.graphics.tsa.plot_acf(arima111.resid, lags=20)
plt.close()
sm.graphics.tsa.plot_pacf(arima111.resid, lags=20)
plt.close()

# Use first half of the observations to predict the second half.
# Not bad! 
print("Suggested model is ARIMA(1,1,1).")
preds = arima111.predict(start=len(diff1)//2+1)
times = range(1,len(diff1)+1)
plt.plot(times[len(diff1)//2:], diff1[len(diff1)//2:], 'b')
plt.plot(times[len(diff1)//2:], preds, 'r')

hand_obs = mlines.Line2D([], [], color="b", label="Observed")
hand_fore = mlines.Line2D([], [], color="r", label="Forecast")
plt.legend(handles=[hand_obs, hand_fore])

plt.title('Second Half of Observations')
plt.xlabel('Time')
plt.ylabel('Hemoglobin Response')

plt.savefig(location_of_images+"ts-preds.png")
