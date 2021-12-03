import os, sys

sys.path.append(os.path.abspath(os.path.join("../")))

import pandas as pd
import numpy as np
import logging
import multiprocessing
from tqdm.auto import tqdm
from helper_libraries.model_pipeline import *

# Start logger
logger = logging.getLogger()


def produce_forecasts_rolling(
    Y,
    X,
    model_list,
    forecasting_pipeline,
    ins_window="30d",
    oos_window="1d",
    expanding=False,
    disable_progress_bar=False,
    parpool=None,
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

    # Function that runs for each iteration t
    # Defining the function as a global is a hack to stop the pool from
    # throwing a "pickle" error
    global iteration_func

    def iteration_func(t):

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
            forecast_log_t = {
                "skip": True,
                "date_ins_start": date_ins_start,
                "date_ins_end": date_ins_end,
                "date_oos_start": date_oos_start,
                "date_oos_end": date_oos_end,
            }
            return t, None, forecast_log_t

        # Perform forecasts
        forecast_output_t, forecast_log_t = forecasting_pipeline(
            model_list, Y_ins, X_ins, Y_oos, X_oos
        )

        # Format results
        forecast_log_t["skip"] = False
        forecast_log_t["date_ins_start"] = date_ins_start
        forecast_log_t["date_ins_end"] = date_ins_end
        forecast_log_t["date_oos_start"] = date_oos_start
        forecast_log_t["date_oos_end"] = date_oos_end

        # Logger
        logger.info(f"Completed iteration {t}")

        return t, forecast_output_t, forecast_log_t

    # # Set up forecast iterator
    # if parpool:
    #     logger.info("Setting up forecast iterator (with multiprocessing)")
    #     iteration_map = parpool.imap_unordered(iteration_func, range(T))
    # else:
    #     logger.info("Setting up forecast iterator (no multiprocessing)")
    #     iteration_map = map(iteration_func, range(T))
    #
    # # Keep forecasting until we run out of OOS data
    # for t, forecast_output_t, forecast_log_t in tqdm(
    #     iteration_map, total=T, disable=disable_progress_bar
    # ):
    #
    #     # Save results
    #     forecast_output[t] = forecast_output_t
    #     forecast_log[t] = forecast_log_t

    # Perform forecasts
    if parpool:
        logging.info("Starting parallel pool")
        with multiprocessing.Pool(parpool) as p:
            for t, forecast_output_t, forecast_log_t in tqdm(
                p.imap_unordered(iteration_func, range(T)),
                total=T,
                disable=disable_progress_bar,
            ):
                # Save results
                forecast_output[t] = forecast_output_t
                forecast_log[t] = forecast_log_t
            logging.info("Shuting down parallel pool")
    else:
        for t, forecast_output_t, forecast_log_t in tqdm(
            map(iteration_func, range(T)),
            total=T,
            disable=disable_progress_bar,
        ):
            # Save results
            forecast_output[t] = forecast_output_t
            forecast_log[t] = forecast_log_t

    return forecast_output, forecast_log
