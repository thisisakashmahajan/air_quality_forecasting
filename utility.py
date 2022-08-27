import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from math import sqrt


# Converts the time series data into a supervised machine learning dataset i.e., N features and 1 output variable
def sequence_to_table(data, look_back=5):
    column_name = 'feature'
    columns = []  # List of column names
    for i in range(look_back):
        # When look_back = 5, 5 different columns names will be generated. For example, feature1, feature2,..., feature5
        columns.append(column_name + str(i + 1))
    features, target = [], []

    for i in range(len(data) - look_back - 1):
        # When look_back = 5, select first 5 records from ith index
        feature = data[i:i + look_back]
        if len(feature) != look_back:
            continue
        # Save the selected records as a separate array
        features.append(feature)
        # The immediate next value of these features would be predicted value
        target.append(data[i + look_back])
    # Create dataframe of the feature
    df = pd.DataFrame(features)
    # Set column names
    df.columns = columns
    # 'next' column is nothing value to be predicted based on selected features
    df['next'] = pd.Series(target)
    return df


def metrics(y, forecast):
    _rmse = sqrt(mse(y, forecast))
    _r2 = r2_score(y, forecast)
    _mae = mae(y, forecast)
    _mape = mape(y, forecast)

    print('RMSE: %.2f' % _rmse)
    print('R2 score: %.2f' % _r2)
    print('MAE: %.2f' % _mae)
    print('MAPE: %.2f' % _mape)


def forecast_for(model, steps, last_record, seq_mean):
    predictions = []
    input = last_record

    for i in range(steps):
        input = input.reshape(1, 1, 30)
        predict = model.predict(input)
        input = input.ravel()
        predictions.append(round(predict[0][0] + seq_mean, 2))
        input = np.delete(input, 0)
        input = np.append(input, predict[0][0])

    return predictions