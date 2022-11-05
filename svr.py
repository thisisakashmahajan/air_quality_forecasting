import warnings

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.svm import SVR
from utility import sequence_to_table

warnings.simplefilter('ignore')

df = pd.read_csv('data/kolkata.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

# seq_mean = np.mean(df.PM25.values)
# df['PM25'] = df['PM25'] - seq_mean

series = sequence_to_table(df['PM25'].values, look_back=30)
X = series.drop(columns=['next']).values
Y = series.next.values

model = SVR(kernel='poly')
model.fit(X, Y)
fitted_series = model.predict(X)

error = sqrt(mean_squared_error(Y, fitted_series))
r2 = r2_score(Y, fitted_series)

print('RMSE:', round(error, 2))
print('R2 score:', round(r2, 2))
