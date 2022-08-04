import pandas as pd
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

df = pd.read_csv('new_data/kolkata.csv', parse_dates=True)
df.set_index('date', inplace=True)  # set the index of time series to date column
df.index.freq = 'D'  # The indices are separated over a day. They are daily records.

model = SimpleExpSmoothing(df.PM25.values).fit(smoothing_level=0.2)

fitted_values = model.fittedvalues
error = mean_absolute_error(df.PM25.values, fitted_values)
print('Error: %.2f' % error)

plt.plot(fitted_values)
plt.show()
