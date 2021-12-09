# Add other paths
import os, sys

sys.path.append(os.path.abspath(os.path.join("../")))

# Should be default
import pandas as pd
import numpy as np
import logging
import logging.config
import warnings
import datetime as dt
import multiprocessing
import glob

# Stuff you need to have on the server
from tqdm.auto import tqdm
from sklearn import linear_model
import seaborn as sns
import sklearn.preprocessing
import sklearn.utils
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Our libraries
from mlmodels.model import *
from mlmodels.linear_models import *
from mlmodels.treebased_models import *
from mlmodels.pca_selection import *
from mlmodels.lasso_selection import *
from helper_libraries.model_pipeline import *
from helper_libraries.forecasting_tools import *
from helper_libraries.preprocessing_tools import *

if __name__ == "__main__":

    ## Params

    # Directory
    main_directory = "../../"

    # Number of processes for parpool
    parpool = len(os.sched_getaffinity(0))

    # Results output files
    results_folder = "data/_temp/"
    forecast_output_filename = (
        main_directory
        + results_folder
        + f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_test_.parquet"
    )
    forecast_log_filename = (
        main_directory
        + results_folder
        + f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_test_log.pkl"
    )

    # Logger
    log_filename = (
        f"{main_directory}code/logs/"
        + f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_test_postselection.log"
    )
    logging.config.fileConfig(
        "../_config/logging.conf", defaults={"log_filename": log_filename}
    )
    logging.info("Starting logger")

    ### Data
    logging.info("Loading data")

    ## Import data
    # Get returns, continuous component, and jump component
    sample_data_df = pd.concat(
        [pd.read_parquet(x) for x in (glob.glob("../../data/proc/_temp/*_cts.parquet"))]
    )
    sample_data_cts_df = pd.concat(
        [pd.read_parquet(x) for x in (glob.glob("../../data/proc/_temp/*_all.parquet"))]
    )
    sample_data_jmp_df = pd.concat(
        [pd.read_parquet(x) for x in (glob.glob("../../data/proc/_temp/*_jmp.parquet"))]
    )

    # Fix column labels
    sample_data_cts_df.columns = [col + "__cts" for col in sample_data_cts_df.columns]
    sample_data_jmp_df.columns = [col + "__jmp" for col in sample_data_jmp_df.columns]

    # Concat the dataframes
    concat_data_df = pd.concat(
        [sample_data_df, sample_data_cts_df, sample_data_jmp_df], axis=1
    )
    logging.info(
        f"Raw data shape: ({len(concat_data_df)}, {len(concat_data_df.T)})",
    )

    # Add date column - needed for adding lagged averages
    concat_data_df["date"] = pd.to_datetime(concat_data_df.index.date)

    ## Define data for forecasting

    # Predict this guy
    Y = concat_data_df[["ff__mkt"]]

    # Use these signals
    #X_factors = ["ff__mkt", "ff__hml", "ff__smb", "ff__rmw", "ff__cma", "ff__umd"]
    #X_factors = sum([[x + "__cts", x + "__jmp"] for x in X_factors], [])
    X_factors  = list(pd.concat([sample_data_cts_df, sample_data_jmp_df], axis=1).columns)
    
    # Add a bunch of lagged averages to use as predictors
    old_columns = concat_data_df.columns
    # Most recent lagged return
    concat_data_df = add_lagged_intradaily_averages(
        concat_data_df, X_factors, window=1, lag_amount=1
    )
    # Average return over the last hour
    concat_data_df = add_lagged_intradaily_averages(
        concat_data_df, X_factors, window=4, lag_amount=1
    )
    # Average return on the last day, this updates each day
    concat_data_df = add_lagged_daily_averages(
        concat_data_df, X_factors, window="1d", lag_amount=1
    )
    new_columns = list(set(concat_data_df.columns).difference(old_columns))

    # Define the signals as new columns added above (no t+1 information)
    X = concat_data_df[new_columns].dropna(axis=0)

    # Fix indices
    new_idx = np.sort(list(set(Y.index).intersection(X.index)))
    Y, X = Y.loc[new_idx], X.loc[new_idx]
    logging.info(
        f"Preprocessed Y shape: ({len(Y)}, {len(Y.T)})",
    )
    logging.info(
        f"Preprocessed X shape: ({len(X)}, {len(X.T)})",
    )
    ### Estimate

    ## Set up estimators
    logging.info("Setting up estimators")

    # Component algos
    model_forecast_ols   = LinearRegression({})
    
    model_forecast_lasso = LASSO(
        {"lambda": 1e-4, "use_intercept": True, "seed": 666}, n_iter = 200
    )
    
    model_forecast_enet  = ENet(
        {"lambda": 1e-4, "l1_ratio": 0.5, "use_intercept": True, "seed": 666}, n_iter = 200
    )
    
    model_forecast_rf = RandomForest(
        {"features": 1, "n_tree": 500, "seed": 5}, n_iter=5, n_signals  = X.iloc[1].shape
    )
    
    model_selection_lasso = LASSO_selection({})
    model_selection_pca = PCA_selection({})

    # Post-selection algos
    model_pca_ols = PostSelectionModel(model_selection_pca, model_forecast_ols)
    model_pca_rf = PostSelectionModel(model_selection_pca, model_forecast_rf)
    model_lasso_ols = PostSelectionModel(model_selection_lasso, model_forecast_ols)
    model_lasso_rf = PostSelectionModel(model_selection_lasso, model_forecast_rf)

    model_list = [
        model_forecast_ols,
        model_forecast_lasso,
        model_forecast_rf
    ]

    ## Main

    # This function is passed into the produce_forecasts_*** function
    # and applied on whatever basis
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

    ## Rolling implementation of models to get rolling forecasts
    logging.info("Producing rolling forecasts")
    forecast_output, forecast_log = produce_forecasts_rolling(
        Y,
        X,
        model_list,
        forecasting_pipeline,
        ins_window="5d",
        oos_window="2d",
        disable_progress_bar=True,
        parpool=parpool,
    )

    ### Export
    logging.info("Preparing results for export")

    ## Formatting
    # Format the forecast results
    forecast_output_df = pd.DataFrame(forecast_output).T.sort_index()
    forecast_output_df.index.name = "t"
    forecast_output_df.columns.name = "model"
    forecast_output_fmt_df = pd.concat(
        [
            pd.concat(forecast_output_df[model_col].values).iloc[:, 0].rename(model_col)
            for model_col in forecast_output_df.columns
        ],
        axis=1,
    )

    # Format the log
    forecast_log_df = pd.DataFrame(forecast_log).T.sort_index()
    forecast_log_df.index.name = "t"

    # Export
    forecast_output_fmt_df.columns = [str(x) for x in forecast_output_fmt_df.columns]
    logging.info("Saving output to " + forecast_output_filename)
    forecast_output_fmt_df.to_parquet(forecast_output_filename)
    logging.info("Saving log to " + forecast_log_filename)
    forecast_log_df.to_pickle(forecast_log_filename)

    logging.info("Done!")
