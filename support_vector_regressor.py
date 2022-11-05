"""
filename: support_vector_regressor.py

This file contains script to forecast using Support Vector Regressor (SVR)
"""
import warnings  # Not necessary. Only imported to filter unwanted interpreter warnings

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score   # Metrics used to evaluate a forecast model. (See methodology section in original paper)
from math import sqrt   # No built-in RMSE. Calculate MSE and take root of it to calculate RMSE
from sklearn.svm import SVR
from utility import sequence_to_table   # See 'utility.py' for more information on this function

warnings.simplefilter('ignore')   # Ignore interpreter warnings, if any. (Try commenting this line!)

df = pd.read_csv('data/kolkata.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

# This step is feature conversion. The mean is subtracted and time series is scaled down. (See Methodology section 3.3 in original paper for more information)
seq_mean = np.mean(df.PM25.values)
df['PM25'] = df['PM25'] - seq_mean

series = sequence_to_table(df['PM25'].values, look_back=30)   # Previous 30 values are used as feature. Hence, each input have 30 feature. (Try altering this value!)
X = series.drop(columns=['next']).values
Y = series.next.values

# The Polynomial kernel is used. Default is 'Radial Basis Function (RBF)'
model = SVR(kernel='poly')
model.fit(X, Y)
fitted_series = model.predict(X)   # Provide the whole series on which the model is trained to get orig

# Calculate errors and print
error = sqrt(mean_squared_error(Y, fitted_series))
r2 = r2_score(Y, fitted_series)

print('RMSE:', round(error, 2))
print('R2 score:', round(r2, 2))
