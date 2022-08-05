import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import numpy as np

warnings.simplefilter('ignore')

df = pd.read_csv('delhi.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

# Single Exponential Smoothing
model = ExponentialSmoothing(endog=df).fit(smoothing_level=0.8)

# Double Exponential Smoothing
# model = ExponentialSmoothing(endog=df, trend='add').fit(smoothing_level=0.8, smoothing_trend=0.2)

# forecast = model.forecast(30)
# print(np.mean(forecast))

# Triple Exponential Smoothing
"""
p = 0.05
model = ExponentialSmoothing(endog=df, trend='add', seasonal='add', damped_trend=True).fit(
        smoothing_level=0.8,
        smoothing_trend=0.2,
        smoothing_seasonal=0.1,
        damping_trend=0.05)
"""

fitted_series = model.fittedvalues
error = sqrt(mean_squared_error(df.PM25.values, fitted_series))
r2 = r2_score(df.PM25.values, fitted_series)
mae = mean_absolute_error(df.PM25.values, fitted_series)

print('Average of series:', round(np.mean(df.PM25.values), 2))
print('Average of damped fitted series:', round(np.mean(fitted_series), 2))
print('Average of the forecasted series:', round(np.mean(model.forecast(30)), 2))
