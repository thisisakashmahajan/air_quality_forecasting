"""
This is support library.

It contains functions to print error rate and display graph.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import pandas as pd


def plot(original, predictions):
    plt.figure().set_dpi(256)
    plt.plot(original, label='Original series', marker='o')
    plt.plot(predictions, label='predictions', marker='o')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('PM2.5 concentration level')
    plt.show()


def print_errors(original, predictions):
    mae = mean_absolute_error(original, predictions)
    rmse = math.sqrt(mean_squared_error(original, predictions))
    print('MAE: %.2f' % mae)
    print('RMSE: %.2f' % rmse)


def sequence_to_table(data, look_back=5):
    """
    :param data: A 1-dimensional Numpy array or list
    :param look_back: The number of values to be considered as features i.e., number of attributes
    :return: DataFrame consist of feature sequence with target timeseries value
    """
    column_name = 'feature'
    columns = []
    for i in range(look_back):
        columns.append(column_name + str(i + 1))
    features, target = [], []

    for i in range(len(data) - look_back - 1):
        feature = data[i:i+look_back]
        if len(feature) != look_back:
            continue
        features.append(feature)
        target.append(data[i+look_back])
    df = pd.DataFrame(features)
    df.columns = columns
    df['next'] = pd.Series(target)
    return df