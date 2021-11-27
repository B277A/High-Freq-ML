import pandas as pd
import numpy as np
import glob
from tqdm.auto import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
import seaborn as sns
import datetime as dt
import concurrent.futures


def resample_data(sret_df, delta_n_arg="15min"):

    ## Factor-cumulativereturns-resampled dataframe
    scr_df = (1 + sret_df).cumprod().reset_index().copy()
    scr_df["date"] = scr_df["datetime"].dt.date

    # Get factor-returns-resampled on a given basis
    srr_df = (
        scr_df.groupby(["date", pd.Grouper(key="datetime", freq=delta_n_arg)])
        .first()
        .pct_change(fill_method=None)
    )

    # First row will be missing returns, add that back in using first row of original dataframe
    srr_df.iloc[0] = srr_df.iloc[0].fillna(sret_df.iloc[0])

    # Clean up index
    srr_df = srr_df.reset_index().drop(["date"], axis=1).set_index("datetime")

    # Convert to log returns: factor-logreturn-resample
    slr_df = np.log(1 + srr_df)
    slrid_df = slr_df.copy()

    return slrid_df


def compute_bp_var(slrid_df):
    # Computes Bipower variation, TOD, and RV

    # Number of days
    n = pd.Series(slrid_df.index).dt.time.nunique()

    # Bipower variation components for each (i,t)
    slrid_bpit_df = np.abs(slrid_df) * np.abs(slrid_df.shift(1))

    # Bipower variation averaged for each i (averaged over t)
    slrid_bpi_df = (
        slrid_bpit_df.reset_index()
        .assign(timeofday=pd.Series(slrid_bpit_df.index).dt.time)
        .groupby("timeofday")
        .mean()
    )

    # Time-of-day correction
    slrid_tod_df = slrid_bpi_df  # / np.mean(slrid_bpi_df, axis=0))

    # Set first intrdaily b to second intrdaily bpi
    slrid_tod_df.loc[slrid_tod_df.index[1]] = slrid_tod_df.loc[slrid_tod_df.index[2]]

    # Compute average
    slrid_tod_df = slrid_tod_df / (slrid_tod_df.iloc[1:, :].sum(axis=0) / (n - 1))

    # Bipower variation (i,t) - drop overnight associated variation
    slrid_bpit_no_df = slrid_bpit_df.copy()
    slrid_bpit_no_df.loc[slrid_bpit_no_df.at_time("09:30:00").index, :] = np.nan
    slrid_bpit_no_df.loc[
        slrid_bpit_no_df.at_time(slrid_df.index[1].time()).index, :
    ] = np.nan

    # Bipower variation daily (t) - ignore overnight associated returns
    slrid_bpt_df = (
        (np.pi / 2)
        * (n - 1)
        / (n - 2)
        * (
            slrid_bpit_no_df.reset_index()
            .assign(date=pd.Series(slrid_df.index).dt.date)
            .drop(["datetime"], axis=1)
            .groupby("date")
            .sum()
        )
    )

    # RV intradaily
    slrid_no_df = slrid_df.copy()
    slrid_no_df.loc[slrid_no_df.at_time("09:30:00").index, :] = np.nan

    # Realized variation - no overnight
    slrid_rvt_df = (
        np.square(slrid_no_df)
        .reset_index()
        .assign(date=pd.Series(slrid_df.index).dt.date)
        .drop(["datetime"], axis=1)
        .groupby("date")
        .sum()
    )

    return slrid_tod_df, slrid_bpt_df, slrid_rvt_df


def identify_jumps(
    slrid_df, slrid_tod_df, slrid_bpt_df, slrid_rvt_df, alpha=4, omegabar=0.49
):

    # Although factor returns, modify to match stock code
    for df in [slrid_df, slrid_tod_df, slrid_bpt_df, slrid_rvt_df]:
        df.columns.name = "permno"

    # Params
    delta_n = pd.Series(slrid_df.head(3).index).diff(1).iloc[1].seconds / (24 * 60 * 60)

    # Create data frame with returns and volatility indicators
    slrid_melt_df = slrid_df.reset_index().melt(
        id_vars=["datetime"], value_name="log_return"
    )
    slrid_melt_df["date"] = slrid_melt_df["datetime"].dt.date
    slrid_melt_df["timeofday"] = slrid_melt_df["datetime"].dt.time
    slrid_melt_df = slrid_melt_df.merge(
        slrid_tod_df.reset_index().melt(
            id_vars="timeofday", value_name="tod_correction"
        ),
        on=["timeofday", "permno"],
        how="left",
    )
    slrid_melt_df = slrid_melt_df.merge(
        slrid_bpt_df.reset_index().melt(id_vars="date", value_name="bipower"),
        on=["date", "permno"],
        how="left",
    )
    slrid_melt_df = slrid_melt_df.merge(
        slrid_rvt_df.reset_index().melt(id_vars="date", value_name="realizedvar"),
        on=["date", "permno"],
        how="left",
    )

    ## Seperate jumps from cts returns
    ## Handle overnight return by scaling alpha
    ## Mark overnight returns as overnight
    ## Can later treat overnight returns seperately or as jumps
    # Threshold for jumps
    slrid_melt_df["cut"] = (
        alpha
        * np.sqrt(slrid_melt_df["tod_correction"] * slrid_melt_df["bipower"])
        * delta_n ** omegabar
    )

    if slrid_melt_df["cut"].min().min() < 0:
        raise Exception("Negative threshold")

    ## Handle overnight returns
    # Ratio between average size of squared log return for overnight returns
    # versus intradaily returns, for each factor
    slrid_melt_df["log_return_sq"] = slrid_melt_df["log_return"] ** 2
    slrid_melt_df["log_return_sq_overnight"] = slrid_melt_df["log_return_sq"] * (
        slrid_melt_df["timeofday"] == dt.time(9, 30)
    )
    slrid_melt_df["log_return_sq_intradaily"] = slrid_melt_df["log_return_sq"] * (
        slrid_melt_df["timeofday"] != dt.time(9, 30)
    )
    slrid_melt_df["avg_var_overnight"] = slrid_melt_df.groupby(["permno"])[
        "log_return_sq_overnight"
    ].transform("mean")
    slrid_melt_df["avg_var_intradaily"] = slrid_melt_df.groupby(["permno"])[
        "log_return_sq_intradaily"
    ].transform("mean")
    slrid_melt_df["var_ratio_overnight_intradaily"] = (
        slrid_melt_df["avg_var_overnight"] / slrid_melt_df["avg_var_intradaily"]
    )

    # Adjust threshold for overnight return by scaling by the average
    # ratio between overnight and intradaily vol
    slrid_melt_df.loc[
        slrid_melt_df["timeofday"] == dt.time(9, 30), "cut"
    ] = slrid_melt_df["cut"] * np.sqrt(slrid_melt_df["var_ratio_overnight_intradaily"])

    # Also mark overnight returns explicitly
    slrid_melt_df["is_overnight"] = 0
    slrid_melt_df.loc[slrid_melt_df["timeofday"] == dt.time(9, 30), "is_overnight"] = 1

    # Identify jumps
    slrid_melt_df["is_jump"] = (
        np.abs(slrid_melt_df["log_return"]) > slrid_melt_df["cut"]
    ).astype(int)

    return slrid_melt_df


def detect_jumps(
    sret_df, alpha=4, omegabar=0.49, skip_resample=False, delta_n_arg="15min"
):

    # Although factor returns, modify to match stock code
    sret_df.columns.name = "permno"

    # Resample data
    if skip_resample:
        slrid_df = sret_df.copy()
    else:
        slrid_df = resample_data(sret_df, delta_n_arg=delta_n_arg)

    # Truncation identification
    slrid_tod_df, slrid_bpt_df, slrid_rvt_df = compute_bp_var(slrid_df)
    slrid_melt_df = identify_jumps(
        slrid_df,
        slrid_tod_df,
        slrid_bpt_df,
        slrid_rvt_df,
        alpha=alpha,
        omegabar=omegabar,
    )

    # Break down of returns
    sret_df = slrid_melt_df[["datetime", "permno", "log_return", "is_jump"]].copy()
    sret_df["log_return_cts"] = sret_df["log_return"] * (1 - sret_df["is_jump"])
    sret_df["log_return_jmp"] = sret_df["log_return"] * sret_df["is_jump"]
    sret_df = sret_df.rename(columns={"permno": "asset"})
    sret_pivot_temp_df = sret_df.set_index("datetime")[
        ["asset", "log_return", "log_return_cts", "log_return_jmp"]
    ].pivot(columns=["asset"])

    ## Pivot
    # Pivotted returns, cts returns, and jump returns
    sret_pivot_df = sret_pivot_temp_df["log_return"].copy()
    sret_pivot_c_df = sret_pivot_temp_df["log_return_cts"].copy()
    sret_pivot_j_df = sret_pivot_temp_df["log_return_jmp"].copy()

    # Fix labels
    sret_pivot_df.columns = [
        ("stock__" if (type(x) == int) else "") + str(x) for x in sret_pivot_df.columns
    ]
    sret_pivot_c_df.columns = [
        ("stock__" if (type(x) == int) else "") + str(x)
        for x in sret_pivot_c_df.columns
    ]
    sret_pivot_j_df.columns = [
        ("stock__" if (type(x) == int) else "") + str(x)
        for x in sret_pivot_j_df.columns
    ]

    # Return results
    return sret_pivot_df, sret_pivot_c_df, sret_pivot_j_df
