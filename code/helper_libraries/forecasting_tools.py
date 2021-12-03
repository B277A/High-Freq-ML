import os, sys

sys.path.append(os.path.abspath(os.path.join("../")))

import pandas as pd
import numpy as np
from helper_libraries.model_pipeline import *
from tqdm.auto import tqdm


def produce_forecasts_rolling(
    Y,
    X,
    model_list,
    forecasting_pipeline,
    ins_window="30d",
    oos_window="1d",
    expanding=False,
    pipeline_kwargs={},
    disable_progress_bar=False,
):

    # Date info
    date_zero = Y.index[0]
    date_stop = Y.index[-1]

    # Output data
    forecast_output = {}
    forecast_log = {}

    # Number of iterations
    # given by ceil((OOS Length)/(OOS Window))
    T = int(
        np.ceil(
            (date_stop - date_zero - pd.Timedelta(ins_window))
            / pd.Timedelta(oos_window)
        )
    )

    # Keep forecasting until we run out of OOS data
    for t in tqdm(range(T), disable=disable_progress_bar):
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
            forecast_log[t] = {
                "skip": True,
                "date_ins_start": date_ins_start,
                "date_ins_end": date_ins_end,
                "date_oos_start": date_oos_start,
                "date_oos_end": date_oos_end,
            }
            continue

        # Perform forecasts and save results
        forecast_output_t, forecast_log_t = forecasting_pipeline(
            model_list, Y_ins, X_ins, Y_oos, X_oos, **pipeline_kwargs
        )
        forecast_output[t] = forecast_output_t
        forecast_log_t["skip"] = False
        forecast_log_t["date_ins_start"] = date_ins_start
        forecast_log_t["date_ins_end"] = date_ins_end
        forecast_log_t["date_oos_start"] = date_oos_start
        forecast_log_t["date_oos_end"] = date_oos_end
        forecast_log[t] = forecast_log_t

    return forecast_output, forecast_log
