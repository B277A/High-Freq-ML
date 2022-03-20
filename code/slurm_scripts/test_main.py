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
from mlmodels.neuralnet import *
from mlmodels.pca_selection import *
from mlmodels.lasso_selection import *
from helper_libraries.model_pipeline import *
from helper_libraries.forecasting_tools import *
from helper_libraries.preprocessing_tools import *
from helper_libraries.data_preprocessing import *

if __name__ == "__main__":
    
    
    
    ij = sys.argv[1]
    ij = ij.split(',')
    i = int(ij[0])
    j = int(ij[1])
    
    drop_overnight  = 'True'
    factors         = 'all'
    lag_setting     = 'HAR'
    setting         = 'main'
    forecast_preiod =  '15m'
    taget_variable = 'ff__mkt'
    model_list      = ['LR','LR_PCA_select', 'LR_Lasso_select', 'Lasso', 'Enet']

    # Number of processes for parpool
 #   parpool = len(os.sched_getaffinity(0))

    main_directory = "../../"

    # Results output files
    results_folder = setting + "/temp_results/"
    forecast_output_filename = (
        main_directory
        + results_folder
        + f"{model_list[i]}_core_{j}_oss.parquet"
    )
    forecast_log_filename = (
        main_directory
        + results_folder
        + f"{model_list[i]}_core_{j}_test_log.pkl"
    )
    insample_output_filename = (
        main_directory
        + results_folder
        + f"{model_list[i]}_core_{j}_insample.parquet"
    )

    # Logger
    log_filename = (
        f"{main_directory}" + setting +"/logs/"
        + f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_test_postselection.log"
    )
    logging.config.fileConfig(
        "../_config/logging.conf", defaults={"log_filename": log_filename}
    )
    logging.info("Starting logger")


    ### Data
    logging.info("Loading data")
    X, Y = data_preprocssing(factors = factors, taget_variable = taget_variable, 
                             drop_overnight = drop_overnight, forecast_preiod = forecast_preiod, lag_setting = lag_setting)
    
    ## Set up estimators
    logging.info("Setting up estimators")


    model_selection_lasso = LASSO_selection({})
    model_selection_pca = PCA_selection({})
    
    if i == 0: # LR
        model_forecast_ols   = LinearRegression({})
        model_list           = [model_forecast_ols]
        
    if i == 1: #LR - PCA
        model_forecast_ols   = LinearRegression({})
        model_pca_ols = PostSelectionModel(model_selection_pca, model_forecast_ols)
        model_list           = [model_pca_ols]
        
    if i == 2: #LR - Lasso
        model_forecast_ols   = LinearRegression({})
        model_lasso_ols = PostSelectionModel(model_selection_lasso, model_forecast_ols)
        model_list           = [model_lasso_ols]
        
    if i == 3: #Lasso
        model_forecast_lasso = LASSO(
        {"lambda": 1e-4, "use_intercept": True, "seed": 666}, n_iter = 100
        )
        model_list           = [model_forecast_lasso]
        

    if i == 4: # Enet
        model_forecast_enet  = ENET(
        {"lambda": 1e-4, "use_intercept": True, "seed": 666, "l1_ratio": 1e-4}, n_iter = 1000
        )
        model_list           = [model_forecast_enet]
        



    ## Rolling implementation of models to get rolling forecasts
    logging.info("Producing rolling forecasts")

    forecast_output, forecast_log, insample_output_t = produce_forecasts_rolling(
        Y,
        X,
        model_list,
        forecasting_pipeline,
        ins_window="730d",
        oos_window="730d",
        expanding=True,
        disable_progress_bar=False,
        parpool=None,
        split_run = True,
        time = j,
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
    insample_output_t = insample_output_t[0]
    logging.info("Saving output to " + insample_output_filename)
    insample_output_t.to_parquet(insample_output_filename)
    logging.info("Saving log to " + forecast_log_filename)
    forecast_log_df.to_pickle(forecast_log_filename)

    logging.info("Done!")
