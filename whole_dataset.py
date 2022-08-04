import pandas as pd
from numpy import mean
from exponential_smoothing import triple_exp_smoothing
from statistics import variance
import matplotlib.pyplot as plt

df = pd.read_csv('delhi.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

forecasts = triple_exp_smoothing(df, alpha=0.8, beta=0.1, gamma=0.1, return_forecast=True)

print('Mean Original:', mean(df.PM25.values))
print('Mean Forecast:', mean(forecasts.values))
print('Variance Original:', variance(df.PM25.values))
print('Variance Forecast:', variance(forecasts.values))

plt.plot(df.PM25.values[-500:], label='Original series')
plt.plot(forecasts.values[-500:], label='Forecasted series')
plt.legend()
plt.show()
