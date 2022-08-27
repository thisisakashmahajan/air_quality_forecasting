# Air Quality Forecasting using LSTM

**Description**:

Traditionally, Exponential Smoothing and ARIMA processes are used to forecast the time
series components. There are many successful examples where ES and ARIMA
has provided state-of-the-art results.

However, increase in variability and non-linearity in data causes these models
to perform poor. Moreover, these methods are designed for short term forecast so they
cannot perform well on large dataset with lots of noise.

Deep Learning can play vital role in this scenario. The deep learning or neural networks
can forecast time series. There is a neural network which is able to process sequences
is Recurrent Neural Network (RNN). RNN and LSTM, the extended version of RNN, can
perform better with non-linear time series data.

In the project above, we have compared following algorithms for forecasting PM2.5 concentration
on 5 air quality dataset collected from Indian cities namely - Delhi, Mumbai, Chennai, Kolkata, and Hyderabad:

1. Simple Exponential Smoothing
2. Double Exponential Smoothing
3. Triple Exponential Smoothing
4. Simple RNN model
5. Long Short-Term Memory (LSTM) model

LSTM outperformed all others compared methods in terms of RMSE.