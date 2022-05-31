import sys, os
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from pandas.tseries.offsets import MonthEnd, YearEnd
import scipy.io
from functools import reduce
import pyarrow as pa
import datetime as dt
import logging


#### Task

taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])

### Logging
log_filename = f'logs/{taskID}.log'
if os.path.exists(log_filename):
    os.remove(log_filename)
logging.basicConfig(filename=log_filename,
                            filemode='a',
                            format="[{asctime}] — [{funcName:12.12}] — [{levelname:<8s} — Line {lineno:4d}]: {message}",
                            style="{",
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


### Print start 
logging.info(f'Starting job: {taskID}')

#### Setup

### Params

crsp_folder = '/hpc/group/dfe/sa400/data_links/CRSP/daily/'
taq_price_folder = '/hpc/group/dfe/sa400/data_links/TAQ/prices/'
output_folder = '../../data/proc/clean_prices/' # This symlinks to the dfe folder
dlret_folder = '/hpc/group/dfe/sa400/data_links/CRSP/dlret/'
me_folder = '/hpc/group/dfe/sa400/data_links/CRSP/me/'

### Import data

## File lists

# Get list of TAQ files
taq_price_files = glob.glob(taq_price_folder + '*.parquet')
taq_price_files_dates = [x.split('/')[-1].split('_')[0].split('.')[0] for x in taq_price_files]
taq_price_files_dates = list(set(taq_price_files_dates))

# Get list of CRSP files
crsp_files = glob.glob(crsp_folder + '*.parquet')

## Stock info
logging.info(f'Loading stock info')

# For reuse
stock_info_df = pd.read_feather('../../data/keys/stock_universe.feather')
stock_info_df['jdate'] = pd.to_datetime(stock_info_df['dt'].dt.date) + MonthEnd(0)
stock_info_df['jdate_ym'] = stock_info_df['jdate'].dt.strftime('%Y%m')

# Read in data
crspmsedelist_df = pd.read_parquet(dlret_folder + 'dlret.parquet')


#### Funcs

### Params

# List of all times to include
freq = 1 # minute
all_times = [
    x.strftime("%H:%M:%S")
    for x in pd.date_range(
        pd.to_datetime("2000-01-01 09:30:00"),
        pd.to_datetime("2000-01-01 16:00:00"),
        freq=f"{freq}min",
    )
]
all_times_taq = all_times

# Need to drop extra data
drop_times = [
    "16:01:00",
    "16:02:00",
    "16:03:00",
    "16:04:00",
    "16:05:00",
]

### Main

def clean_crsp(yyyymm):
    
    # Read in data
    try:
        crsp_df = pd.concat([pd.read_parquet(x) for x in crsp_files if f'/{yyyymm}' in x])
    except:
        logging.warning(f'Using alternative data loading method for {yyyymm}')
        crsp_df = pd.concat([pd.read_parquet(x) for x in crsp_files if f'/{yyyymm}' in x],
                           ignore_index = True)

    # Clean up columns
    crsp_df.columns = [x.lower() for x in crsp_df.columns]
    crsp_df[['ret', 'retx']] = crsp_df[['ret', 'retx']].apply(pd.to_numeric, errors = 'coerce')

    # Fix dates
    crsp_df['date'] = pd.to_datetime(crsp_df['date'], format = '%Y%m%d')

    # Get close and open prices
    crsp_df['prc'] = np.abs(crsp_df['prc'])
    crsp_df['openprc'] = np.abs(crsp_df['openprc']).fillna(crsp_df['prc'])
    
    # Catch nonsensical open prices: rule is to find cases where close price is 
    # 5 times larger than open price while close-to-close return  is less than 
    # 90% in magnitude
    crsp_df = crsp_df.sort_values(by = 'date')
    crsp_df['prc_lead'] = crsp_df.groupby(['permno'])['prc'].shift(-1)
    crsp_df.loc[
        crsp_df.query("prc/openprc > 5 & abs(ret) < 0.90").index, "openprc"
    ] = crsp_df.loc[crsp_df.query("prc/openprc > 5 & abs(ret) < 0.90").index, "prc"]

    # Infer close-to-open adjusted overnight returns 
    crsp_df['ret_open_close_intraday'] = (crsp_df['prc']-crsp_df['openprc'])/crsp_df['openprc']
    crsp_df['ret_close_open_adj'] = (1+crsp_df['ret'])/(1+crsp_df['ret_open_close_intraday']) - 1
    crsp_df['retx_close_open_adj'] = (1+crsp_df['retx'])/(1+crsp_df['ret_open_close_intraday']) - 1
    
    # Add lagged market equity
    crsp_me_subset_df = pd.read_parquet(f'{me_folder}{yyyymm}.parquet')
    crsp_df = crsp_df.merge(crsp_me_subset_df, on = ['permno', 'date'], how = 'left')
    
    # Add additional stock info
    stock_info_subset_df = stock_info_df.loc[stock_info_df['jdate_ym'] == yyyymm]
    proc_df = crsp_df.merge(
        stock_info_subset_df[[ 'permno', 'jdate', 'shrcd', 'exchcd', 'gvkey', 'mcap_rank']],
        on=["permno"],
        how="left",
    )
        
    # Filter by share/exchange code and whether stock is primary
    proc_df = proc_df.query('shrcd in (10,11) & exchcd in (1,2,3)')
    
    # Hand-cleaning, no easy way to deal with these 'bad' permnos, so just drop
    proc_df = proc_df.loc[~proc_df['permno'].isin([90806, 83712, 47387])]
    if int(yyyymm[:4]) < 2018:
        proc_df = proc_df.loc[~proc_df['permno'].isin([15293])]
    if int(yyyymm) == 200301:
        proc_df = proc_df.loc[~proc_df['permno'].isin([87658])]
        
    # Create dataframes for start and end of the day
    crsp_df_start = proc_df.copy()
    crsp_df_end = proc_df.copy()
    crsp_df_end = crsp_df_end.merge(crspmsedelist_df, on = ['date', 'permno'], how = 'left')

    crsp_df_start['time'] = '09:30:00'
    crsp_df_end['time'] = '16:00:00'

    crsp_df_start['price'] = crsp_df_start['openprc']
    crsp_df_end['price'] = crsp_df_end['prc']

    crsp_df_start['return'] = crsp_df_start['ret_close_open_adj']
    crsp_df_end['return'] = (1+crsp_df_end['ret_open_close_intraday'].fillna(0))*(1+crsp_df_end['dlret'].fillna(0))-1

    crsp_df_start['returnx'] = crsp_df_start['retx_close_open_adj']
    crsp_df_end['returnx'] = crsp_df_end['ret_open_close_intraday']
    
    crsphf_df = pd.concat([crsp_df_start, crsp_df_end], ignore_index = True)
    
    # Add datetime info
    crsphf_df['datetime'] = crsphf_df['date'] + pd.to_timedelta(crsphf_df['time'])
    crsphf_df['yyyymm'] = crsphf_df['date'].dt.strftime('%Y%m').astype(int)    
    
    return crsphf_df[['datetime', 'date', 'permno', 'cusip', 'price', 'return', 'returnx', 'dlret', 'meq_close_lag', 'me_close_lag']]

def clean_taq(date):

    # Get TAQ files
    taq_df = pd.read_parquet(
        taq_price_folder + date + ".parquet", columns=["permno", "symbol", "date", "time", "price"]
    )

    # Drop "when issued" shares
    taq_df = (
        taq_df.assign(symbol_last_2=taq_df["symbol"].str.slice(-2))
        .query('symbol_last_2 != "WI"')
        .drop("symbol_last_2", axis=1)
        .copy()
    )

    # Drop cases with missing permnos
    taq_df = taq_df.loc[taq_df["permno"].str.isnumeric()]
    taq_df["permno"] = pd.to_numeric(taq_df["permno"]).astype(int)

    # Fix "9:..:.." instead of "09:..:.."
    taq_df.loc[taq_df['time'].str.len() < 8, 'time'] = '0' + taq_df.loc[taq_df['time'].str.len() < 8, 'time']
    
    # Drop extra times
    taq_df = taq_df.query('time not in @drop_times')

    # Handle any missing times
    index = pd.MultiIndex.from_product(
        [taq_df["permno"].unique(), all_times_taq], names=["permno", "time"]
    )
    index_df = pd.DataFrame(index=index).reset_index()
    taq_df = (
        taq_df.merge(index_df, on=["permno", "time"], how="right")
        .sort_values(by=["permno"])
        .astype({"time": "category"})
    )
    taq_df = taq_df.sort_values(by=["permno", "time"])
    
    # Catch scenario where there are less than full set of 
    # prices in day (Thanksgiving, Christmas, etc)
    valid_times = taq_df[['time', 'price']].dropna(axis=0)['time'].unique()
    if len(valid_times) < len(all_times):
        logging.warning(f'[{date}] This date has only {len(valid_times)} valid times')

    taq_df = taq_df.loc[taq_df['time'].isin(list(valid_times))]

    # Forward fill in entries
    ffill_cols = ["price"]  # 'cusip9', 'symbol', 'ticker_identifier'
    taq_df[ffill_cols] = taq_df.groupby(["permno"])[ffill_cols].ffill()
    taq_df["date"] = int(date)

    # Add date
    taq_df["datetime"] = pd.to_datetime(taq_df["date"], format="%Y%m%d") + pd.to_timedelta(
        taq_df["time"].astype(str)
    )
    
    # Drop cases where realized quarticity for the day is extremely high, just a 
    # crude way of catching very bad data
    taq_df = taq_df.sort_values(by=["permno", "datetime"])
    taq_df['price_pct_change'] = taq_df.groupby(['permno'])['price'].transform('pct_change').fillna(0)
    taq_df["log_ret_4"] = np.power(np.log(1 + taq_df["price_pct_change"]), 4)
    taq_df = taq_df[
        ~taq_df["permno"].isin(
            pd.DataFrame(taq_df.groupby(["permno"])["log_ret_4"].sum() > 1).query("log_ret_4").index
        )
    ]

    # Sort
    taq_df = taq_df.sort_values(by=["permno", "datetime"]).reset_index(drop=True)

    return taq_df[["permno", "datetime", "price"]].drop_duplicates()

def add_value_weights(rm_df):

    # Add value weights
    rm_df = rm_df.sort_values(by=["permno", "datetime"]).reset_index(drop=True)
    rm_df["1+retx"] = rm_df["returnx"].fillna(0) + 1
    rm_df["cumretx"] = rm_df.groupby(["permno", "date"])["1+retx"].cumprod()
    rm_df["meq_close_lag_times_cumretx"] = rm_df["meq_close_lag"] * rm_df["cumretx"]
    rm_df["me_close_lag_times_cumretx"] = rm_df["me_close_lag"] * rm_df["cumretx"]
    rm_df["value_wt"] = rm_df.groupby(["permno"])["meq_close_lag_times_cumretx"].shift(1)
    rm_df["value_wt"] = rm_df["value_wt"].fillna(rm_df["meq_close_lag"])
    rm_df["value_wt_permno"] = rm_df.groupby(["permno"])["me_close_lag_times_cumretx"].shift(1)
    rm_df["value_wt_permno"] = rm_df["value_wt"].fillna(rm_df["me_close_lag"])
    rm_df = rm_df.drop(
        [
            "1+retx",
            "cumretx",
            "meq_close_lag_times_cumretx",
            "meq_close_lag",
            "me_close_lag_times_cumretx",
            "me_close_lag",
            "price",
            "crsp_taq_merge_indicator",
        ],
        axis=1,
    )

    return rm_df


def filter_bad_merges(rm_df, close_hour, close_minute):
    """
    Deals with potentially incorrect merges between TAQ and CRSP
    by checking if the prices make sense. Main check is to see if
    the intradaily prices jump far too much at the open and close.
    """

    ## Deal with TAQ and CRSP mismatches
    # Coarse procedure to catch mismatched TAQ and CRSP stocks
    rm_df = rm_df.sort_values(by=["permno", "datetime"])
    rm_df["price_pct_change"] = rm_df.groupby(["permno"])["price"].pct_change()
    rm_df["price_pct_change_abs"] = np.abs(rm_df["price_pct_change"])
    rm_df.loc[
        (rm_df["datetime"].dt.time == dt.time(9, 35))
        | (rm_df["datetime"].dt.time == dt.time(close_hour, close_minute)),
        "price_pct_change",
    ] = np.nan
    rm_df["price_vol"] = rm_df.groupby(["permno"])["price_pct_change"].transform("std")

    # State which stocks need to be dropped
    drop_df = rm_df.loc[
        (rm_df["price_pct_change_abs"] ** 2 > 3 * rm_df["price_vol"])
        & (rm_df["price_pct_change_abs"] > 0.05)
        & (rm_df["meq_close_lag"] > 1e5)
    ]
    drop_instances_df = (
        drop_df.groupby(["permno"])["datetime"]
        .count()
        .rename("instances")
        .astype(int)
        .reset_index()
        .query("instances == 2")
    )
    if len(drop_instances_df):
        logging.warning(
            f'[{rm_df.iloc[0]["date"].date()}] Dropping PERMNOs {", ".join(drop_instances_df["permno"].unique().astype(str))} '
            + "due to potential mismatches"
        )
    rm_df = rm_df.drop(["price_pct_change", "price_pct_change_abs", "price_vol"], axis=1)

    # Drop intradaily prices for those stocks - interpolation will fill in missing later
    rm_df.loc[
        (rm_df["datetime"].dt.time != dt.time(9, 30))
        & (rm_df["datetime"].dt.time != dt.time(close_hour, close_minute))
        & (rm_df["permno"].isin(drop_instances_df["permno"].unique())),
        "price",
    ] = np.nan

    return rm_df


def merge_crsp_taq(date, clean_crsp_date_df, interpolate=True):

    # Store additional info
    log = [date]

    # Get clean TAQ data
    clean_taq_df = clean_taq(date.strftime("%Y%m%d"))

    # Handle any missing times
    all_datetimes = pd.to_datetime(clean_taq_df["datetime"].unique())
    index = pd.MultiIndex.from_product(
        [clean_crsp_date_df["permno"].unique(), all_datetimes], names=["permno", "datetime"]
    )
    index_df = pd.DataFrame(index=index).reset_index()

    # Adjust CRSP close time to match close time based on TAQ data
    close_hour = np.max(all_datetimes).hour
    close_minute = np.max(all_datetimes).minute

    if (close_hour != 16) or (close_minute != 0):
        logging.warning(
            f"[{date.date()}] Adjusting market close time to (H={close_hour}, M={close_minute})"
        )
        clean_crsp_date_df.loc[
            clean_crsp_date_df["datetime"] == clean_crsp_date_df["datetime"].max(), "datetime"
        ] = clean_crsp_date_df.loc[
            clean_crsp_date_df["datetime"] == clean_crsp_date_df["datetime"].max(), "datetime"
        ].apply(
            lambda x: x.replace(hour=close_hour, minute=close_minute)
        )

    # Adjust CRSP open time to match close time based on TAQ data
    open_hour = np.min(all_datetimes).hour
    open_minute = np.min(all_datetimes).minute

    if (open_hour != 9) or (open_minute != 30):
        logging.warning(
            f"Adjusting market open time for {date.date()} to (H={open_hour}, M={open_minute})"
        )
        clean_crsp_date_df.loc[
            clean_crsp_date_df["datetime"] == clean_crsp_date_df["datetime"].min(), "datetime"
        ] = clean_crsp_date_df.loc[
            clean_crsp_date_df["datetime"] == clean_crsp_date_df["datetime"].min(), "datetime"
        ].apply(
            lambda x: x.replace(hour=open_hour, minute=open_minute)
        )
        raise NotImplementedError(
            f"Market open is delayed on {date.date()} to (H={open_hour}, M={open_minute}); "
            + "Adjusting for this problem has not been implemented"
        )

    # Resample the CRSP data
    resample_df = clean_crsp_date_df.merge(index_df, on=["permno", "datetime"], how="right")

    # Merge with taq
    rm_df = resample_df.merge(
        clean_taq_df,
        on=["permno", "datetime"],
        how="left",
        suffixes=["_crsp", "_taq"],
        indicator="crsp_taq_merge_indicator",
    )
    rm_df["crsp_taq_merge_indicator"] = pd.Categorical(rm_df["crsp_taq_merge_indicator"])

    # Create merged price series
    rm_df["price"] = rm_df["price_crsp"].fillna(rm_df["price_taq"])
    rm_df = rm_df.drop(["price_crsp", "price_taq"], axis=1)

    ## Fix missing data
    # Fill in missing columns
    rm_df["date"] = date
    rm_df["meq_close_lag"] = rm_df.groupby(["permno"])["meq_close_lag"].ffill()
    rm_df["me_close_lag"] = rm_df.groupby(["permno"])["me_close_lag"].ffill()
    rm_df["cusip"] = rm_df.groupby(["permno"])["cusip"].ffill()

    ## Deal with mismatches between TAQ and CRSP
    rm_df = filter_bad_merges(rm_df, close_hour, close_minute)

    ## Interpolate log prices and fill in missing prices/returns
    # otherwise just use standard forward filling
    if interpolate:
        rm_df["log_price"] = np.log(rm_df["price"])
        rm_df["log_price_last"] = rm_df.groupby(["permno"])["log_price"].transform("last")
        rm_df["log_price_first"] = rm_df.groupby(["permno"])["log_price"].transform("first")
        rm_df["interp_beta"] = (rm_df["log_price_last"] - rm_df["log_price_first"]) / (
            len(all_datetimes) - 1
        )
        rm_df["count"] = rm_df.groupby(["permno"])["datetime"].cumcount()
        rm_df["log_price_interp"] = rm_df["log_price_first"] + rm_df["count"] * rm_df["interp_beta"]
        rm_df["price"] = rm_df["price"].fillna(np.exp(rm_df["log_price_interp"]))
        rm_df = rm_df.drop(
            [
                "log_price",
                "log_price_last",
                "log_price_first",
                "interp_beta",
                "count",
                "log_price_interp",
            ],
            axis=1,
        )
    else:
        rm_df["price"] = rm_df.groupby(["permno"])["price"].ffill()
        rm_df["price"] = rm_df.groupby(["permno"])["price"].bfill()

    # Add returns
    rm_df["return"] = np.where(
        rm_df["datetime"].dt.time.astype(str) == "09:30:00",
        rm_df["return"],
        rm_df.groupby(["permno"])["price"].pct_change(),
    )
    rm_df["returnx"] = np.where(
        rm_df["datetime"].dt.time.astype(str) == "09:30:00",
        rm_df["returnx"],
        rm_df.groupby(["permno"])["price"].pct_change(),
    )
    # Fix last return for delisting
    rm_df["return"] = (1 + rm_df["return"]) * (1 + rm_df["dlret"].fillna(0)) - 1
    rm_df = rm_df.drop(["dlret"], axis=1)

    # Any remaining missing returns will be those originally missing from CRSP
    rm_df["return"] = rm_df["return"].fillna(0)

    # Add value-weights
    rm_df = add_value_weights(rm_df)
    
    # Drop duplicates - really shouldn't be any
    n_prev = len(rm_df)
    rm_df = rm_df.drop_duplicates(['datetime', 'date', 'permno'])
    if len(rm_df) < n_prev:
        logging.warning(
            f"Dropped duplicates for {date.date()}; observations went from {n_prev} to {len(rm_df)}"
        )

    return rm_df

def helper_func(df_group):
    return merge_crsp_taq(df_group[0], df_group[1], interpolate = True)

def process_date(yyyymm):

    # Clean CRSP dataframe for the month
    clean_crsp_df = clean_crsp(yyyymm)
    logging.info(f'Read CRSP Data for {yyyymm}')

    # Go through each date and infill TAQ prices
    for df in map(helper_func, clean_crsp_df.groupby(["date"])):
        date_str = df['datetime'].iloc[0].strftime("%Y%m%d")
        df.to_parquet(output_folder + date_str + '.parquet')
        logging.info(f'Processed {date_str}')
    
    return


#### Processing 

# List of dates
yyyymm_list = np.sort(
    list(map(lambda x: pd.to_datetime(x).strftime("%Y%m"), stock_info_df.query('dt >= "1996"')["dt"].unique()))
)

# Get task
yyyymm = yyyymm_list[taskID]

# Perform
logging.info(f'Beginning task of processing {yyyymm}')
process_date(yyyymm)