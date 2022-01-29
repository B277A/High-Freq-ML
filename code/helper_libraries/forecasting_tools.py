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

def forecasting_pipeline(model_list, Y_ins, X_ins, Y_oos, X_oos):

    # Output variables
    forecast_log = {}

    # Train the algos
    mtrain = ModelTrainer(model_list, Y_ins, X_ins, seed=444)
    mtrain.validation()
    forecast_log["hyperparameters"] = mtrain.model_hyperparameters_opt

    # Produce OOS forecasts
    mtest = ModelTester(mtrain)
    oos_forecasts, model_params = mtest.forecast(Y_oos, X_oos)
    forecast_output = oos_forecasts
    forecast_log["fitted_parameters"] = model_params

    return forecast_output, forecast_log
    
def produce_forecasts_rolling(
    Y,
    X,
    model_list,
    forecasting_pipeline,
    ins_window="1000d",
    oos_window="1d",
    expanding=False,
    disable_progress_bar=False,
    parpool=None,
    time = None,
    split_run = None,
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

        print(t)
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

        return t, forecast_output_t, forecast_log_t
    
    global in_sample_func

    def in_sample_func():


        # Define data
        Y_ins = Y
        X_ins = X
        Y_oos = Y
        X_oos = X


        # Perform forecasts
        insample_output_t, forecast_log_t = forecasting_pipeline(
            model_list, Y_ins, X_ins, Y_oos, X_oos
        )


        return insample_output_t
    
        # Perform forecasts
    if parpool:

        logging.info(f"Starting multiprocessing pool with {parpool} processes")
        with multiprocessing.Pool(parpool) as p:

            for t, forecast_output_t, forecast_log_t in tqdm(
                p.imap_unordered(iteration_func, range(T)),
                total=T,
                disable=disable_progress_bar,
            ):

                # Save results
                forecast_output[t] = forecast_output_t
                forecast_log[t] = forecast_log_t

                # Logger
                logger.info(f"Completion Progress: {len(forecast_output.keys())}/{T}")

            logging.info("Shuting down parallel pool")

    elif split_run:
         logger.info("Split run")
         t, forecast_output_t, forecast_log_t = iteration_func(time)
         # Save results
         forecast_output[t] = forecast_output_t
         forecast_log[t] = forecast_log_t
         
         logger.info(f"Completion Progress: {time}/{T}")
        
    else:
        
        for t, forecast_output_t, forecast_log_t in tqdm(
            map(iteration_func, range(T)),
            total=T,
            disable=disable_progress_bar,
        ):
            
            # Save results
            forecast_output[t] = forecast_output_t
            forecast_log[t] = forecast_log_t
            
            logger.info(f"Completion Progress: {len(forecast_output.keys())}/{T}")
            
        
    insample_output_t = in_sample_func()
    logger.info("In Sample estimate")

    return forecast_output, forecast_log, insample_output_t
