"""
filename: distribution.py

This file contains scripts to:

 1. Show count of invalid PM2.5 records (not in range between 0 and 500, inclusive)
 2. Show Auto-correlation plot for a time series

"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf   # 'statsmodels' must be installed in your environment. 'plot_acf' stands for Plot results of Auto-correlation function


df = pd.read_csv("data/hyderabad.csv", parse_dates=['date'])

# Two 'len' functions simply counts the rows where either PM2.5 is greater than 500 or is less than 0 (marked invalid).
print("Invalid values:", len(df[df['PM25'] > 500]) + len(df[df['PM25'] < 0]))

plt.figure(figsize=(5, 5)).set_dpi(512)

# 'plot_acf' requires two important parameters - the time series and the number of observations for which correlation is to be calculated
plot_acf(df.PM25.values, lags=30, title=None)

# Y-limit is set to avoid larger figure and unnecessary coordinate (Try to comment the following line to see effect!)
plt.ylim(-0.25, 1.10)
plt.show()
