import os, sys

sys.path.append(os.path.abspath(os.path.join("../")))

import pandas as pd
import numpy as np
from helper_libraries.model_pipeline import *


def rolling_pre(Y, X, model_list, ins_window="30d", oos_window="1d"):

    # Initial data
    date_zero = Y.index[0]
    date_stop = Y.index[-1]
    date_oos_start = date_zero

    # Output data
    num_models = len(model_list)
    model_forecasts_list = [[] * num_models]
    log = {}
    log["hyperparameters"] = []

    # Iteration
    t = 0

    # Keep forecasting until we run out of OOS data
    while date_oos_start < date_stop:
        #         print('Iteration: ', t)

        # Define dates
        date_ins_start = date_zero + t * pd.Timedelta(oos_window)
        date_ins_end = date_ins_start + pd.Timedelta(ins_window) - pd.Timedelta("1s")
        date_oos_start = date_ins_start + pd.Timedelta(ins_window)
        date_oos_end = date_oos_start + pd.Timedelta(oos_window) - pd.Timedelta("1s")

        # Define data
        Y_ins = Y.loc[date_ins_start:date_ins_end, :]
        X_ins = X.loc[date_ins_start:date_ins_end, :]
        Y_oos = Y.loc[date_oos_start:date_oos_end, :]
        X_oos = X.loc[date_oos_start:date_oos_end, :]

        # No data for this OOS period, so just skip
        if not len(Y_oos):
            t += 1
            continue

        mtrain = ModelTrainer(model_list, Y_ins, X_ins, seed=444)
        mtrain.validation()
        log["hyperparameters"].append(mtrain.model_hyperparameters_opt)

        mtest = ModelTester(mtrain)
        oos_forecasts = mtest.forecast(Y_oos, X_oos)

        for i in range(len(model_list)):
            oos_forecast = oos_forecasts[i]
            oos_forecast.index = Y_oos.index
            model_forecasts_list[i].append(oos_forecast)

        t += 1

    return model_forecasts_list


def produce_forecasts_rolling(
    Y, X, model_list, forecasting_pipeline, ins_window="30d", oos_window="1d", expanding = False, pipeline_kwargs={}
):

    # Date info
    date_zero = Y.index[0]
    date_stop = Y.index[-1]

    # Output data
    forecast_output = {}
    forecast_log = {}

    # Number of iterations
    # given by ceil((OOS Data)/(OOS Window))
    T = int(np.ceil((date_stop - date_zero - pd.Timedelta(ins_window)) / pd.Timedelta(oos_window)))

    # Keep forecasting until we run out of OOS data
    for t in range(T):
        # print('Iteration: ', t)

        # Define dates
        date_ins_start = date_zero + t * pd.Timedelta(oos_window)
        date_ins_end = date_ins_start + pd.Timedelta(ins_window) - pd.Timedelta("1s")
        date_oos_start = date_ins_start + pd.Timedelta(ins_window)
        date_oos_end = date_oos_start + pd.Timedelta(oos_window) - pd.Timedelta("1s")
        
        # If we are using an expanding training window
        if expanding:
            date_ins_start = date_zero

        # Define data
        Y_ins = Y.loc[date_ins_start:date_ins_end, :]
        X_ins = X.loc[date_ins_start:date_ins_end, :]
        Y_oos = Y.loc[date_oos_start:date_oos_end, :]
        X_oos = X.loc[date_oos_start:date_oos_end, :]

        # No data for this OOS period, so just skip
        if not len(Y_oos):
            forecast_log[t] = {"skip": True}
            continue

        # Perform forecasts and save results
        forecast_output_t, forecast_log_t = forecasting_pipeline(
            model_list, Y_ins, X_ins, Y_oos, X_oos, **pipeline_kwargs
        )
        forecast_output[t] = forecast_output_t
        forecast_log[t] = forecast_log_t
        forecast_log[t]["skip"] = False

    return forecast_output, forecast_log