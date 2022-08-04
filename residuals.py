from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from statistics import variance

df = pd.read_csv('new_data/kolkata.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

print(df[-30:])
