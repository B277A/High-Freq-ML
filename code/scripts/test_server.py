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
    main_directory = "../.."

    # Logger
    log_filename = (
        f"{main_directory}/code/logs/"
        + f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_test_postselection.log"
    )
    logging.config.fileConfig(
        "../_config/logging.conf", defaults={"log_filename": log_filename}
    )

    ### Data

    # Set up some sample data for the tests
    sample_data_df = pd.read_parquet("../../data/proc/_temp/1996_all.parquet")

    ## Define data for forecasting
    Y = sample_data_df[["ff__mkt"]].head(2500)
    X = (
        sample_data_df[["ff__hml", "ff__smb", "ff__rmw", "ff__cma", "ff__umd"]]
        .shift(1)
        .bfill()
    ).head(2500)

    ### Estimate

    ## Set up estimators
    logging.info("Setting up estimators")

    # Component algos
    model_forecast_lasso = LASSO(
        {"lambda": 1e-5, "use_intercept": False, "seed": 5}, n_iter=12
    )
    model_forecast_enet = ENet(
        {"lambda": 1e-5, "l1_ratio": 0.5, "use_intercept": True, "seed": 5}, n_iter=14
    )
    model_forecast_ols = LinearRegression({})
    model_selection_pca = PCA_selection({})
    model_selection_lasso = LASSO_selection({})

    # Post-selection algos
    model_pca_lasso = PostSelectionModel(
        model_selection_pca, model_forecast_lasso, n_iter=58
    )
    model_lasso_ols = PostSelectionModel(model_selection_lasso, model_forecast_ols)
    model_pca_enet = PostSelectionModel(model_selection_pca, model_forecast_enet)
    model_list = [
        model_forecast_lasso,
        model_pca_lasso,
        model_lasso_ols,
        model_pca_enet,
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

    # Start parallel pool

    ## Rolling implementation of models to get rolling forecasts
    logging.info("Producing rolling forecasts")
    forecast_output, forecast_log = produce_forecasts_rolling(
        Y,
        X,
        model_list,
        forecasting_pipeline,
        ins_window="60d",
        oos_window="1d",
        disable_progress_bar=False,
        parpool=12,
    )

    # Shut down parallel pool

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
    forecast_log_df.columns.name = "model"

    # Export
    forecast_output_fmt_df.columns = [str(x) for x in forecast_output_fmt_df.columns]
    forecast_output_fmt_df.to_parquet(f"{main_directory}/data/_temp/test.parquet")
    forecast_log_df.to_pickle(f"{main_directory}/data/_temp/test_log.pkl")

    logging.info("Done!")
