# warnings are imported to supress unwanted warnings
import warnings

# calculates square root of number - used to calculate RMSE
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Evaluation metrics to evaluate the performance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Exponential Smoothing class
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.simplefilter('ignore')

df = pd.read_csv('data/kolkata.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

seq_mean = np.mean(df.PM25.values)
df['PM25'] = df['PM25'] - seq_mean

# Single Exponential Smoothing
# model = ExponentialSmoothing(endog=df).fit(smoothing_level=0.8)

# Double Exponential Smoothing
# model = ExponentialSmoothing(endog=df, trend='add').fit(smoothing_level=0.8, smoothing_trend=0.2)

# forecast = model.forecast(30)
# print(np.mean(forecast))

# Triple Exponential Smoothing
p = 0.5  # This value represents damping factor for exponential smoothing
model = ExponentialSmoothing(endog=df, trend='add', seasonal='add', damped_trend=False).fit(
        smoothing_level=0.8,
        smoothing_trend=0.2,
        smoothing_seasonal=0.1,
        damping_trend=p)

fitted_series = model.fittedvalues  # Exponential smoothing calculates new fitted values

# Calculate error metrics
error = sqrt(mean_squared_error(df.PM25.values, fitted_series + seq_mean))
r2 = r2_score(df.PM25.values, fitted_series + seq_mean)
mae = mean_absolute_error(df.PM25.values, fitted_series + seq_mean)

# print('RMSE:', round(error, 2))
# print('R2 score:', round(r2, 2))
# print('MAE:', round(mae, 2))

# Next 15 days forecasts can be retrieved
forecast = model.forecast(60)

for i in forecast.values:
    print(i + seq_mean)

plt.plot(np.append(df.PM25.values[-50:], forecast), label='Forecasted series')
plt.plot(df.PM25.values[-50:], label='Original series', color='black')
plt.legend()
# plt.show()
