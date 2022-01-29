# Add other paths
import os, sys

sys.path.append(os.path.abspath(os.path.join("../")))

import glob
import numpy as np
import pandas as pd
import logging

from helper_libraries.preprocessing_tools import *

def data_preprocssing(factors = 'all',
        taget_variable = 'market', overnight = 'True'):
    
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
    if taget_variable == 'market':
        Y = concat_data_df[["ff__mkt"]]
    
    if factors == 'all':
        X_factors  = list(pd.concat([sample_data_cts_df, sample_data_jmp_df], axis=1).columns)
    
    elif factors == 'fama_6':
        X_factors = ["ff__mkt", "ff__hml", "ff__smb", "ff__rmw", "ff__cma", "ff__umd"]
        
    elif factors == 'benchmark':
        X_factors = ["ff__mkt"]
        

    if overnight == 'False':
        temp_drop = []

        for i in range(0,len(concat_data_df)):
            if str(concat_data_df.iloc[:,0].index[i]).split()[1] == '09:30:00':
                temp_drop.append(concat_data_df.index[i])
        
        concat_data_df = concat_data_df.drop(temp_drop)

    
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
    
    

    return X, Y
    
