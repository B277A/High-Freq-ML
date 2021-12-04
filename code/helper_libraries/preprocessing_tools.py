import pandas as pd
import numpy as np

def add_lagged_intradaily_averages(data_df, columns, window=None, lag_amount=None, win_type=None):
    """
    Computes a rolling average over the given window of the high-frequency returns 
    and shifts forward by the given lag amount. 
    
    Parameters
    ----------
    data_df : DataFrame
        The DataFrame to add columns to. This function does modifies the original
        dataframe and also returns it. 
    columns : list
        Columns in the dataframe to target. 
    window : string
        Length of window for the rolling average; should be something like
        '1h' or '90min' for one-hour or ninety-minute averages; note that the 
        observation itself counts itself in the rolling window, so the observation 
        that is {lag_amount} elements after the overnight return will usually 
        only include the overnight return (unless the window is larger than the 
        overnight period)
    lag_amount : int
        Observations to lag the rolling average by. Generally, should be 
        set to 1. 
    win_type : str
        Passed to the pandas rolling function, can be any string accepted
        by the scipy.signal window function
        
    Returns
    -------
    DataFrame
        An updated version of the data_df input dataframe. New columns 
        are labelled as "{inputcolumn}_{window}_{lag_amount}_lagged_intradaily_avg"
        
    Examples
    --------
    Add rolling one-hour average of high-frequency returns lagged by one observation
    >>> data_sample_df = add_lagged_intradaily_averages(
    >>>   data_sample_df, ["ff__mkt", "ff__smb"], "1h", 1
    >>> )
    """

    lagged_averages_df = (
        data_df[columns].rolling(window, win_type=win_type).mean().shift(lag_amount)
    )

    for col in columns:
        data_df[col + f"_{window}_{lag_amount}_lagged_intradaily_avg"] = lagged_averages_df[
            col
        ]

    return data_df

def add_lagged_daily_averages(data_df, columns, window=None, lag_amount=None, win_type=None):
    """
    First computes the average return for each day in the dataset. Then,
    computing a rolling average over the given window of the daily returns 
    and shifts forward by the given lag amount. 
    
    Parameters
    ----------
    data_df : DataFrame
        The DataFrame to add columns to. This function does not modify 
        the dataframe directly. 
    columns : list
        Columns in the dataframe to target. 
    window : string
        Length of window for the rolling average; should be something like
        '1d' or '22d' for daily or monthy averages; note that these are 
        trading days not calendar days
    lag_amount : int
        Trading days to lag the rolling average by. Generally, should be 
        set to 1. 
    win_type : str
        Passed to the pandas rolling function, can be any string accepted
        by the scipy.signal window function
        
    Returns
    -------
    DataFrame
        An updated version of the data_df input dataframe. New columns 
        are labelled as "{inputcolumn}_{window}_{lag_amount}_lagged_daily_avg"
        
    Examples
    --------
    Add rolling 22-day average of daily returns lagged by one day
    >>> data_sample_df = add_lagged_daily_averages(
    >>>   data_sample_df, ["ff__mkt", "ff__smb"], "22d", 1
    >>> )
    """
    # Make sure date is a column in data_df

    lagged_averages_df = (
        data_df[columns]
        .groupby(pd.Grouper(freq="1d"))
        .mean()
        .rolling(window, win_type=win_type)
        .mean()
        .dropna()
        .shift(lag_amount)
        .reset_index()
        .rename(columns={"datetime": "date"})
    )

    data_df = data_df.merge(
        lagged_averages_df,
        on="date",
        how="left",
        suffixes=("", f"_{window}_{lag_amount}_lagged_daily_avg"),
    ).set_index(data_df.index)
    
    return data_df