import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pymannkendall import original_test
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error


def get_portion_size(portion, amount):
    return int(round(portion * amount))


def get_average(portion, x):
    amount = get_portion_size(portion, len(x))
    return round(np.average(x[:amount]), 2)


def time_series_test(x):
    adf_score = adfuller(x)
    mk_score = original_test(x)
    return adf_score[1], mk_score.p


def sequence_to_table(data, look_back=5):
    column_name = 'feature'
    columns = []
    for i in range(look_back):
        columns.append(column_name + str(i + 1))
    features, target = [], []

    for i in range(len(data) - look_back - 1):
        feature = data[i:i + look_back]
        if len(feature) != look_back:
            continue
        features.append(feature)
        target.append(data[i + look_back])
    df = pd.DataFrame(features)
    df.columns = columns
    df['next'] = pd.Series(target)
    return df


def simple_exp_smoothing(time_series, alpha=0.1, forecast_for=30, return_forecast=False):
    """
    :param time_series: A Pandas dataframe containing time series
    :param alpha: smoothing level
    :param forecast_for: Number of days to forecast ahead
    :param return_forecast: Whether to return the forecasted values
    :return: mean absolute error of forecasts
    """
    model = SimpleExpSmoothing(time_series[:-forecast_for]).fit(smoothing_level=alpha)
    forecasts = model.forecast(forecast_for)
    if return_forecast:
        return forecasts
    return mean_absolute_error(time_series[-forecast_for:], forecasts.values)


def double_exp_smoothing(time_series, alpha=0.1, beta=0.1, forecast_for=30, return_forecast=False):
    """
    :param time_series: A Pandas dataframe containing time series
    :param alpha: smoothing level
    :param beta: smoothing slope
    :param forecast_for: Number of days to forecast ahead
    :param return_forecast: Whether to return the forecasted values
    :return: mean absolute error of forecasts
    """
    model = Holt(time_series[:-forecast_for]).fit(smoothing_level=alpha, smoothing_slope=beta)
    forecasts = model.forecast(forecast_for)
    if return_forecast:
        return forecasts
    return mean_absolute_error(time_series[-forecast_for:], forecasts.values)


def triple_exp_smoothing(time_series, alpha=0.1, beta=0.1, gamma=0.1,
                         phi=0.1, forecast_for=30, return_forecast=False):
    """
    :param time_series: A Pandas dataframe containing time series
    :param alpha: smoothing level
    :param beta: smoothing slope
    :param gamma: smoothing seasonal
    :param phi: damping slope
    :param forecast_for: Number of days to forecast ahead
    :param return_forecast: Whether to return the forecasted values
    :return: mean absolute error of forecasts
    """
    model = ExponentialSmoothing(time_series,
                                 trend='add', seasonal='add').fit(smoothing_level=alpha,
                                                                  smoothing_slope=beta,
                                                                  smoothing_seasonal=gamma,
                                                                  damping_slope=phi)
    forecasts = model.predict(start=0, end=len(time_series)-1)
    if return_forecast:
        return forecasts
    else:
        return mean_absolute_error(time_series.values, forecasts.values)
