import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('new_data/hyderabad.csv', parse_dates=True)
df.set_index('date', inplace=True)

df.plot()
plt.show()
