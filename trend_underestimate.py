# proof to the incorrect trend factor estimation by DES and TES
import pandas as pd
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def print_on_line(array):
    for item in array:
        print(item)


warnings.filterwarnings("ignore")
df = pd.read_csv('data/kolkata.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

des_model = ExponentialSmoothing(endog=df, trend='add').fit(smoothing_level=0.8, smoothing_trend=0.2)
tes_model = ExponentialSmoothing(endog=df, trend='add', seasonal='add').fit(
        smoothing_level=0.8,
        smoothing_trend=0.2,
        smoothing_seasonal=0.1)

des_trend = des_model.trend
tes_trend = tes_model.trend

original_trend = []
for i in range(1, len(df)):
    diff = df.iloc[i]['PM25'] - df.iloc[i - 1]['PM25']
    original_trend.append(diff)

print('Original trend:')
print_on_line(original_trend[-20:])
print('-----')
print('DES trend:')
print_on_line(des_trend[-20:].values)
print('-----')
print('TES trend:')
print_on_line(tes_trend[-20:].values)
print('-----')
