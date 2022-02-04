import os
import pandas as pd
import numpy as np
import glob
import datetime as dt
from numba import jit

## Params

# Folders
if "Saketh" in os.getcwd():
    hfzoo_data_folder = "../../../../GitHub/HFZoo/data"
    hfzoo_return_folder = f"{hfzoo_data_folder}/proc/factor_returns_with_jumpind/hfzoo"
else:
    hfzoo_data_folder = "/Users/au515538/Desktop/HFML/data"
    hfzoo_return_folder = f"{hfzoo_data_folder}/proc/_temp"

### Main

## Data prep
def load_model_results(
    folder=None,
    overnight=False,
    oos_periods=2,
    predictors="All",
    debug=False,
    models_select=None,
):
    """
    Loads the model results from a particular folder based on the specified arguments.

    Args:
      folder: the folder where the models are stored; using this will ignore the other arguments
      overnight: if True, loads data that includes the overnight period. Defaults to False
      oos_periods: selects models with the given variable value of out-of-sample period length in years. Defaults to 2
      predictors: which predictors were used for forecasting
      debug: if True, will print out the chosen model. Defaults to False
      models_select: a list of models to load. If None, all models are loaded. .

    Returns:
      Two dataframes: forecast_oss_df and forecast_ins_df.
    """

    # Spreadsheet containing a key for all the results folders
    key_df = pd.read_excel("../../results/key.xlsx")

    # If we haven't manually specified a folder
    if not folder:
        key_subset_df = key_df.loc[
            (key_df["Overnight"] == overnight)
            & (key_df["OOSPeriods"] == oos_periods)
            & (key_df["Predictors"] == predictors)
        ]
        if len(key_subset_df) > 1:
            print(key_subset_df)
            raise Exception(
                "More than one set of results matching input parameters have been detected"
            )
        elif len(key_subset_df) == 0:
            print(key_subset_df)
            raise Exception("No model results matching the given parameters detected")
        else:
            folder = "../../results/" + key_subset_df.iloc[0]["Folder"] + "/Results/"

    if debug:
        print("Loading the following model results..." + "\n" + "-" * 50)
        print(key_subset_df.iloc[0].to_string())
        print("-" * 50)

    # Get list of models
    unique_models = np.unique(
        [
            "_".join(x.split("/")[-1].split("_")[:-1])
            for x in glob.glob(f"{folder}/*.parquet")
            if "rvol" not in x and len(x) > 0
        ]
    )

    # If you want to pick particular models
    if models_select:
        # Ensure that all selected models are available in the data
        if not np.all(
            [model_select in unique_models for model_select in models_select]
        ):
            print("[Available models]: " + ", ".join(unique_models))
            print(
                "[Missing models]:   "
                + ", ".join(
                    [
                        model_select
                        for model_select in models_select
                        if model_select not in unique_models
                    ]
                )
            )
            raise Exception("The above selected models are not available in the data")

    ## Load results

    # Initialize dataframe with index from benchmark model
    forecast_oss_df = pd.DataFrame(
        [], index=pd.read_parquet(f"{folder}/Benchmark_oss.parquet").index
    )
    forecast_ins_df = pd.DataFrame(
        [], index=pd.read_parquet(f"{folder}/Benchmark_insample.parquet").index
    )

    # Load data from each of the other models
    for model in unique_models:
        forecast_oss_df["oss_" + model] = pd.read_parquet(
            f"{folder}/" + model + "_oss.parquet"
        ).values
        forecast_ins_df["ins_" + model] = pd.read_parquet(
            f"{folder}/" + model + "_insample.parquet"
        ).values

    return forecast_oss_df, forecast_ins_df


def load_mkt_rf_returns(overnight=True):
    """
    Loads the market returns and the Fama-French risk-fre rate (1 month TBill)
    """

    ### market
    fret_df = pd.concat(
        [
            pd.read_parquet(x, columns=["ff__mkt"])
            for x in glob.glob(f"{hfzoo_return_folder}/*_all.parquet")
        ]
    )

    ### Risk-free
    # Read in data
    ff_daily_df = pd.read_csv(
        "../../data/ff/F-F_Research_Data_5+1_daily.csv", skiprows=3, engine="python"
    )

    # Clean up
    ff_daily_df.columns = ["date", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
    ff_daily_df["date"] = pd.to_datetime(ff_daily_df["date"].astype(str))

    # Select data for risk-free rate
    rf_df = ff_daily_df[["date", "rf"]].query('date > "1995-12-31"').copy()
    rf_df["rf"] = np.log(1 + rf_df["rf"] / 100)
    rf_df["date"] = pd.to_datetime(rf_df["date"])
    rf_df = rf_df.set_index("date")

    # Resample the risk free at high-freq
    hrf_df = (
        fret_df[[]]
        .assign(date=fret_df.index.date)
        .reset_index()
        .set_index("date")
        .join(rf_df)
        .reset_index()
    )
    hrf_df["rf"] = hrf_df["rf"] / hrf_df.groupby("date")["rf"].transform("count")
    hrf_df = hrf_df.set_index("datetime")

    # Drop overnight?
    if not overnight:
        fret_df = fret_df.loc[fret_df.index.time != dt.time(9, 30)]
        hrf_df = hrf_df.loc[hrf_df.index.time != dt.time(9, 30)]

    return fret_df, hrf_df


def load_spreads():

    spread_df = pd.read_csv("../../data/taq/spy_spread.csv")
    spread_df["date"] = pd.to_datetime(spread_df["date"], format="%Y%m%d")
    spread_df = spread_df.set_index("date").resample("15min").first().ffill()

    return spread_df


## Trading strategies
# Inputs are a dictionary with info about the model
# Includes model predictions
def get_weights_passive(model_info_dict):
    """Invest everything in the market."""

    # Just set weights to 1
    # Reuse prediction series to figure out index
    model_pred = model_info_dict["model_pred"]
    weights = model_pred * 0 + 1

    return weights


def get_weights_sign(model_info_dict):
    """Invest based on sign of forecast."""

    # Simply the sign of the predictions
    model_pred = model_info_dict["model_pred"]
    weights = np.sign(model_pred)

    return weights


def get_weights_sign_positive(model_info_dict):
    """Invest based on sign but no shorting"""

    # Sign of the predictions with negative
    # weights replaced with 0
    weights = get_weights_sign(model_info_dict)
    weights[weights < 0] = 0

    return weights


def get_weights_stderr(model_info_dict, cutoff):
    """Invest based on some standard error cutoff.
    Weights are non-negative, cutoff is a arg
    """

    # Invest whenever prediction exceeds cutoff
    model_pred = model_info_dict["model_pred"]
    model_pred_std = model_info_dict["model_pred_std"]
    weights = model_pred > cutoff * model_pred_std

    return weights


def get_weights_tanh(model_info_dict):
    """Invest with weights based on tanh of the
    z-score of the prediction; force positive weights
    """

    # Invest whenever prediction exceeds cutoff
    model_pred = model_info_dict["model_pred"]
    model_pred_std = model_info_dict["model_pred_std"]
    # y = (intradaily sharpe) * (annualization factor)
    y = model_pred / model_pred_std * np.sqrt(252 * 27)
    # z = standardized y
    z = y / np.std(y)
    # tanh of z
    weights = np.tanh(z)
    weights[model_pred < 0] = 0

    return weights


@jit
def compute_ms_weights_fast(trade_ind, cutoff):
    """Trade indicator could either be tanh(zscores)
    or just the sign of the model. The former
    is meant for the "0.5" and "0.9" style strategies,
    while the latter can be used to implement the
    "MS Positive" strategy by setting the cutoff to 0
    """

    n = np.shape(trade_ind)[0]
    weights = np.zeros(n)
    count_post = 0
    in_trade = 0

    for i in range(n):
        if trade_ind[i] > cutoff:
            in_trade = 1

        if trade_ind[i] < 0:
            in_trade = 0

        if in_trade == 1:
            # While we are holding,
            # the weight is set to one
            weights[i] = 1

    return weights


def get_weights_ms_strat(model_info_dict, cutoff):
    """Mathias's top secret trading strategy"""

    # Tanh of z-scores of predictions
    model_pred = model_info_dict["model_pred"]
    # model_pred_std = model_info_dict["model_pred_std"]
    model_pred_z_tanh = np.tanh(model_pred / np.std(model_pred))

    # Get weights using fast numba function
    X_tanh = model_pred_z_tanh.values
    weights = compute_ms_weights_fast(X_tanh, cutoff)
    weights = pd.Series(weights, index=model_pred.index)

    return weights


def get_trading_results(
    forecast_oss_df,
    spread_df,
    market_returns,
    riskfree_returns,
    strategies_list,
    model_list,
    drop_overnight=False,
    hold_cash=True,
):
    """
    Takes the model predictions, the market returns, the riskfree rate, the spread, and the trading
    strategies, and it outputs the results of the trading strategies.
    
    Args:
      forecast_oss_df: The OSS forecast dataframe
      spread_df: The spread dataframe
      market_returns: The market returns as a pd.Series
      riskfree_returns: The riskfree rate of return as a pd.Series
      strategies_list: a list of strings, each of which is a trading strategy name.
      model_list: list of model names
      drop_overnight: whether to drop overnight returns. Defaults to False
      hold_cash: whether to hold cash or not. Defaults to True
    
    Returns:
      oss_results_all_df: a dataframe with the summarized results of each strategy
      oss_returns_all_df: a dataframe with the returns of each strategy
      oss_weights_all_df: a dataframe with the weights of each strategy
      oss_retpred_all_df: a dataframe with the predicted returns of each model
    """

    # Useful variables
    n_years = (forecast_oss_df.index.max() - forecast_oss_df.index.min()).days / 365

    # Output
    oss_results_list = []
    oss_returns_list = []
    oss_weights_list = []
    oss_retpred_list = []

    # Main loop
    for model_idx, model_name in enumerate(model_list):

        # Storing results for each model
        oss_results_model = pd.DataFrame([], index=strategies_list)

        for strategy_idx, strategy_name in enumerate(strategies_list):

            # Basic info
            model_name = model_list[model_idx]
            model_col_name = "oss_" + model_name

            # Model predictions
            model_pred = forecast_oss_df[model_col_name]
            # model_pred_std = np.std(model_pred)
            model_pred_std = (
                pd.read_parquet("../../Results/RV/RV_HAR_oss.parquet")
                .pipe(np.sqrt)
                .iloc[:, 0]
                .loc[model_pred.index]
            )

            # Put into dictionary
            model_info_dict = {}
            model_info_dict["model_pred"] = model_pred
            model_info_dict["model_pred_std"] = model_pred_std
            model_info_dict["spread"] = spread_df["QSpreadPct_TW_m"].loc[
                model_pred.index
            ]
            model_info_dict["riskfree"] = riskfree_returns

            # Get trading strategy results
            if strategy_name == "Market":
                weights = get_weights_passive(model_info_dict)
            elif strategy_name == "Sign":
                weights = get_weights_sign(model_info_dict)
            elif strategy_name == "Positive":
                weights = get_weights_sign_positive(model_info_dict)
            elif strategy_name == "Tanh":
                weights = get_weights_tanh(model_info_dict)
            elif strategy_name == "MS Strategy 0.5":
                weights = get_weights_ms_strat(model_info_dict, 0.5)
            elif strategy_name == "MS Strategy 0.9":
                weights = get_weights_ms_strat(model_info_dict, 0.9)
            else:
                raise Exception("Unknown trading strategy: ", strategy_name)

            # Convert to float
            weights = weights.astype(float)

            # Strategy statistics
            portfolio_returns = weights * market_returns + (1 - weights) * (
                riskfree_returns
            ) * (1 - hold_cash)
            portfolio_turnover = weights.diff(1).fillna(0).abs()

            # Radj = R - |w_diff|*tau
            portfolio_returns_adj = portfolio_returns - portfolio_turnover.to_frame(
                "turnover"
            ).join(spread_df[["QSpreadPct_TW_m"]] / 2).prod(axis=1)

            # Averages
            portfolio_average_return = portfolio_returns.sum() / n_years
            portfolio_average_return_adj = portfolio_returns_adj.sum() / n_years
            portfolio_average_excess_return = (
                portfolio_returns - riskfree_returns
            ).sum() / n_years
            portfolio_average_excess_return_adj = (
                portfolio_returns_adj - riskfree_returns
            ).sum() / n_years
            portfolio_average_turnover = portfolio_turnover.sum() / n_years
            portfolio_average_vol = np.sqrt(
                np.square(portfolio_returns).sum() / n_years
            )
            portfolio_sharpe = portfolio_average_excess_return / portfolio_average_vol
            portfolio_sharpe_adj = (
                portfolio_average_excess_return_adj / portfolio_average_vol
            )

            # Save results to dataframe
            oss_results_model.loc[strategy_name, "Return"] = round(
                portfolio_average_return, 2
            )
            oss_results_model.loc[strategy_name, "ReturnAdj"] = round(
                portfolio_average_return_adj, 2
            )
            oss_results_model.loc[strategy_name, "Trades"] = round(
                portfolio_average_turnover, 1
            )
            oss_results_model.loc[strategy_name, "Sharpe"] = round(portfolio_sharpe, 2)
            oss_results_model.loc[strategy_name, "SharpeAdj"] = round(
                portfolio_sharpe_adj, 2
            )
            oss_results_model.loc[strategy_name, "rvol"] = round(
                portfolio_average_vol, 2
            )
            oss_results_model.loc[strategy_name, "Name (col)"] = model_col_name
            oss_results_model.loc[strategy_name, "Name"] = model_name
            oss_returns_model = portfolio_returns
            oss_returns_model.name = (model_name, strategy_name)
            oss_weights_model = weights
            oss_weights_model.name = (model_name, strategy_name)

            # Append to running list of returns
            oss_returns_list.append(oss_returns_model)
            oss_weights_list.append(oss_weights_model)

        # Model predictions
        oss_retpred_model = model_pred
        oss_retpred_model.name = model_name
        oss_retpred_list.append(oss_retpred_model)

        # Append to running list of results
        oss_results_list.append(oss_results_model)

    # Merge results
    oss_results_all_df = pd.concat(oss_results_list)
    oss_returns_all_df = pd.concat(oss_returns_list, axis=1)
    oss_weights_all_df = pd.concat(oss_weights_list, axis=1)
    oss_retpred_all_df = pd.concat(oss_retpred_list, axis=1)

    return (
        oss_results_all_df,
        oss_returns_all_df,
        oss_weights_all_df,
        oss_retpred_all_df,
    )

