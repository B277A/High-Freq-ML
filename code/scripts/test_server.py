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

    # Some data that comes with seaborn
    iris_df = sns.load_dataset("iris")
    iris_df.head()

    # Add a whole bunch of columns that are rotations of the original data plus noise
    np.random.seed(1)
    iris_hf_df = iris_df.copy()
    K = 150
    X_new_cols = []
    for k in range(K):
        new_col_names = [f"petal_width_{k}", f"sepal_length_{k}", f"sepal_width_{k}"]
        iris_hf_df[new_col_names] = (
            iris_hf_df[["petal_width", "sepal_length", "sepal_width"]]
            @ np.random.rand(3, 3)
            + (np.random.rand(len(iris_hf_df), 3) - 0.5) * 30
        )
        X_new_cols.append(new_col_names)

    # Redefine Y variable as a rotation of the true variables
    iris_hf_df["petal_length"] = iris_hf_df[
        ["petal_width", "sepal_length", "sepal_width"]
    ] @ np.random.rand(3, 1)
    X_new_cols = sum(X_new_cols, [])

    # Add a datetime index
    iris_hf_df.index = pd.date_range(
        start=dt.datetime.today().date(), periods=len(iris_hf_df)
    )

    ## Define data for forecasting
    Y = iris_hf_df[["petal_length"]]
    X = iris_hf_df[X_new_cols]

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

    forecast_output[3][0][0].to_csv(f"../../data/output/test_output_0.csv")

    logging.info("Done!")
