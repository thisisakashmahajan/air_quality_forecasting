import pandas as pd

# timeseries_dataset_from_array converts a numpy array into a supervised learning dataset (feature and target)
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from math import sqrt

df = pd.read_csv('delhi.csv', parse_dates=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

# Extract only the series to be predicted
data = df.PM25.values

# sequence length can be interpreted as number of feature or how many past values should be used to predict next one
seq_length = 10
input_data = data[:-seq_length].reshape((-1, 1))
targets = data[seq_length:].reshape((-1, 1))
dataset = timeseries_dataset_from_array(input_data, targets, sequence_length=seq_length)

inputs = None
targets = None
for batch in dataset:
    inputs, targets = batch

# Neural network model based on single LSTM layer
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(10, 1)))  # Produces a trend effect
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

print('Training...')
history = model.fit(inputs, targets, epochs=100, verbose=False)

feature = np.array(inputs[0])
predictions = []

for i in range(30):
    feature = feature.reshape(1, 10, 1)
    prediction = model.predict(feature)
    feature = feature.ravel()
    feature = np.delete(feature, 0)
    feature = np.append(feature, prediction[0][0])
    predictions.append(prediction[0][0])

print(predictions)
print('Fitting error RMSE:', sqrt(history.history['loss'][-1]))
