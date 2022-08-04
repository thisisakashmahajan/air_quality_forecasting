import pandas as pd
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy as np

df = pd.read_csv('delhi.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

data = df.PM25.values
input_data = data[:-10].reshape((-1, 1))
targets = data[10:].reshape((-1, 1))
dataset = timeseries_dataset_from_array(input_data, targets, sequence_length=10)

inputs = None
targets = None
for batch in dataset:
    inputs, targets = batch

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(10, 1)))  # Produces a trend effect
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(inputs, targets, epochs=100, verbose=True)

feature = np.array(inputs[0])

for i in range(100):
    feature = feature.reshape(1, 10, 1)
    prediction = model.predict(feature)
    feature = feature.ravel()
    feature = np.delete(feature, 0)
    feature = np.append(feature, prediction[0][0])
    print(targets[i][0], prediction[0][0])
