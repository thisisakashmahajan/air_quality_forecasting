import pandas as pd
from utility import get_average

df = pd.read_csv('new_data/kolkata.csv', parse_dates=True)
df.set_index('date', inplace=True)

series = df.PM25.values
average = get_average(1, series)
print(average)
