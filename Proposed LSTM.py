import pandas as pd
from utility import sequence_to_table, metrics, forecast_for
from numpy import mean
import tensorflow as tf
from keras.models import Sequential

df = pd.read_csv('data/delhi.csv', parse_dates=True)
df.set_index('date')

seq_mean = mean(df.PM25.values)

sequence = df.PM25.values - seq_mean
data = sequence_to_table(sequence, look_back=30)

# divide data into feature and target
X = data.drop(columns=['next']).values
y = data.next.values
# first reshape the data
X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))

# Proposed LSTM model
tf.random.set_seed(42)

model = Sequential()
model.add(tf.keras.layers.LSTM(units=512,
                               activation='relu',
                               input_shape=(1, X_reshaped.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(units=512,
                               activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mse', optimizer='adam')

forecast = model.predict(X_reshaped)
metrics(y, forecast)

forecast = forecast_for(model, 10, X_reshaped[-1], seq_mean)
for i in forecast:
    print(i)
