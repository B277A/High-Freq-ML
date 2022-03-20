# Add other paths
import os, sys

sys.path.append(os.path.abspath(os.path.join("../")))

import glob
import numpy as np
import pandas as pd
import logging
import datetime

from helper_libraries.preprocessing_tools import *

def data_preprocssing(factors = 'all',
        taget_variable = 'market', drop_overnight = 'True', forecast_preiod = '15m', lag_setting = 'HAR'):
    
    
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
    concat_data_df["time"] = concat_data_df.index.time
    
    ## Define data for forecasting
    
    # Predict this guy
    Y_factors = [taget_variable]
    
    if factors == 'all':
        X_factors  = list(pd.concat([sample_data_cts_df, sample_data_jmp_df], axis=1).columns)
        
    if factors == 'combined':
        X_factors  = list(sample_data_df.columns)
        
        
    if factors == 'exclude_cts':
        X_factors  = list(sample_data_jmp_df.columns)
    
    if factors == 'exclude_jmp':
        X_factors  = list(sample_data_cts_df.columns)
    
    elif factors == 'fama_6':
        X_factors = ["ff__mkt", "ff__hml", "ff__smb", "ff__rmw", "ff__cma", "ff__umd"]
        
    elif factors == 'benchmark':
        X_factors = ["ff__mkt"]

    
    if drop_overnight == 'True':
        concat_data_df = concat_data_df[concat_data_df.time != datetime.time(9, 30)]
        
    # Add a bunch of lagged averages to use as predictors
    old_columns = concat_data_df.columns
    
    if forecast_preiod == '15m':
        Y = concat_data_df[Y_factors]
    
    if forecast_preiod == '1h':
        Y = pd.concat([concat_data_df[Y_factors].rolling(4, win_type=None)
                       .mean().shift(-3).copy(), concat_data_df.time], axis = 1)
        Y = Y[(Y.time < datetime.time(15, 15)) == True]
        Y = Y.drop(['time'], axis = 1)
  

    if forecast_preiod == '1d':
        Y = (concat_data_df[Y_factors].groupby(pd.Grouper(freq="1d"))
        .mean().rolling('1d', win_type=None).mean().dropna())
        
        Y["date"] = pd.to_datetime(Y.index.date)
    
   
    
    if forecast_preiod != '1d':
        
        if lag_setting  == 'A1':
        # Most recent lagged return
            concat_data_df = add_lagged_intradaily_averages(
            concat_data_df, X_factors, window=1, lag_amount=1
        )
        
        
        elif lag_setting  == 'A4':
                concat_data_df = add_lagged_intradaily_averages(
                    concat_data_df, X_factors, window=1, lag_amount=1
                )
                concat_data_df = add_lagged_intradaily_averages(
                        concat_data_df, X_factors, window=1, lag_amount=2
                    )
                concat_data_df = add_lagged_intradaily_averages(
                                    concat_data_df, X_factors, window=1, lag_amount=3
                                )
                concat_data_df = add_lagged_intradaily_averages(
                                    concat_data_df, X_factors, window=1, lag_amount=4
                                )
        else:
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
        
        new_idx = np.sort(list(set(Y.index).intersection(X.index)))
        Y, X = Y.loc[new_idx], X.loc[new_idx]
    
    if forecast_preiod == '1d':
        X = add_lagged_averages_daily_forecast(
        concat_data_df, X_factors, Y, window="1d", lag_amount=1)
        
        X = add_lagged_averages_daily_forecast(
        concat_data_df, X_factors, X, window="5d", lag_amount=1)
                
        X = add_lagged_averages_daily_forecast(
        concat_data_df, X_factors, X, window="22d", lag_amount=1)
        
        X = X.drop(['date', Y_factors[0]], axis = 1).dropna()
        Y = Y.drop(['date'], axis = 1).iloc[1: , :]
                

    # Fix indices

    logging.info(
        f"Preprocessed Y shape: ({len(Y)}, {len(Y.T)})",
    )
    logging.info(
        f"Preprocessed X shape: ({len(X)}, {len(X.T)})",
    )
    

    return X, Y
    
