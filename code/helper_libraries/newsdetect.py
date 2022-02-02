import pandas as pd
import numpy as np
import glob
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import seaborn as sns
import datetime as dt
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import textwrap
from fuzzywuzzy import process, fuzz
import logging

# Start logger
logger = logging.getLogger()

### Basic functions
# These functions help with loading and filtering the news data

# Load preprocessed DJN data for some month
def get_djn_data(event_datetime, data_dir = '../../data'):

    # Get data for some month
    ym = event_datetime.strftime("%Y-%m")
    djn_ym_df = pd.read_parquet(f"{data_dir}/dowjones/preproc/{ym}.parquet")
    djn_ym_df["datetime"] = (
        djn_ym_df["display_date"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    )

    return djn_ym_df


def find_associated_news(
    news_df, news_time, window="15min", delta="15min", debug=False
):
    """
    Given a news dataframe, a news datetime, and a window, find all news in that window.
    
    Args:
      news_df: the dataframe containing all news articles
      news_time: the time of the news event
      window: the time window in which to search for news. Defaults to 15min. Defaults to 15min.
      delta: the return delta to use when searching for news. Defaults to 15min. Defaults to 15min.
      debug: whether to print debugging statements. Defaults to False. Defaults to False
    
    Returns:
      A dataframe with the news articles that are associated with the given time.
    """

    if dt.time(9, 30) < news_time.time():
        news_in_window = news_df.loc[
            news_df["datetime"].between(
                news_time - pd.Timedelta(window) - pd.Timedelta(delta),
                news_time + pd.Timedelta(window),
            )
        ]
    else:
        start_window = (
            pd.to_datetime(news_time.date()) - pd.Timedelta("1d") + pd.Timedelta("16h")
        )
        end_window = pd.to_datetime(news_time.date()) + pd.Timedelta("9.5h")
        news_in_window = news_df.loc[
            news_df["datetime"].between(start_window, end_window)
        ]

    return news_in_window


# Neatly print out news
def _bordered(text):
    """
    Prints a string with a border around it.
    
    Args:
      text: The text to be displayed.
    
    Returns:
      The function _bordered() is returning a string.
    """
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ["┌" + "─" * (width + 2) + "┐"]
    for s in lines:
        res.append("│ " + (s + " " * width)[:width] + " │")
    res.append("└" + "─" * (width + 2) + "┘")
    return "\n".join(res)


def print_news(news_items_df):
    """
    The function takes in a dataframe of news items and prints out the news items in a readable format.
    
    Args:
      news_items_df: a dataframe of news items
    
    Returns:
      None
    """

    for _, news_item in news_items_df.iterrows():

        print_text = []
        # print_text.append("=" * 144)
        print_text.append(f"[Headline]:     {news_item['headline'].strip()}")
        print_text.append(f"[Display Time]: {news_item['datetime']}")
        print_text.append("\n")
        print_text.append("[Text]:          ")
        print_text.append(
            "\n".join(
                [
                    "\n".join(textwrap.wrap(x, width=100))
                    for x in news_item["text"].strip().split("\n")
                ]
            )
        )
        # print_text.append("=" * 144, end="\n\n")
        print(_bordered("\n".join(print_text)))


# Filter out news not of interest
filters_product = ["P/ORCS", "P/WSP"]
filters_subject = [
    "N/COR",
    "N/SPO",
    "N/SPT",
    "N/NDX",
    "N/TAB",
    "N/DTA",
    "N/COC",
    "N/SGR",
    "N/NYH",
    "N/SVR",
    "N/COR",
    "N/TPCT",
    "N/CTN",
    "N/CTL",
    "N/GRN",
    "N/OSD",
    "N/WHT",
    "N/CAL",
    "N/SDP",
    "N/FTH",
    "N/MCOX",
    "N/ANL",  # Analyst ratings like for bonds
    "N/13D",  # Type of form
    "N/LIF",  # Lifestyles
]
filters_stat = ["S/ACT", "S/DJA"]
filters_headline = [
    "Benchmark Govt Yields",
    "Week Highs And Lows",
    # "MARKET TALK",
    "Toronto Most Actives",
    "Political, Economic Calendar",
    # "Top Stories Of The Day",
    "Most Actives At",
    "OTC Bulletin Board",
    "CBOE Most Active Call, Put",
    "Pacific Exchange Most Active Call, Put",
    "Canada Bill Auction",
    "NYSE Stock Transactions",
    "Table Of Recent Stock Buyback Announcements",
    "1st Quarter Losses",
    "2nd Quarter Losses",
    "3rd Quarter Losses",
    "4th Quarter Losses",
    "New York Foreign Exchange Indications",
    "N.Y. Commodities Futures Settlement Prices",
    "Asian Economic And Political Calendar",
    "Asian Economic/Political/Corporate Calendar",
    "Hollywood",
    "USDA Central Kansas Terminal and Processor Daily Grain Report",
    "Livestock Futures",
    "Global Commodities Roundup",
    "Closing Stock Prices",
    "Ethanol Plant Report",
    "Cash Grain Close",
    "Peru Poll: ",
    "South African Reserve Bank",
    "CME Livestock And Meat Futures Settlement Prices",
    "European Stocks: Stoxx 50 Late Prices",
    "Peru's General Closing Index",
    "FFCB ",
    "This Is A Test From Dow Jones",
    "NJ Rail",
    "NJ Transit",
    "N.J. Transit Train",
    "Nuclear Reactor",
    "Precious Metals Futures Opening",
    "Treasury Coupon Prices Early Table",
    "Treasury Coupon Prices Midday Table",
    "Treasury Coupon Prices Late Table",
    "Nymex Natural Gas Futures",
    "From Hottrend.com",
    "New York Money Market Rate Indications",
    "Chile Stks",
    "Chile stocks",
    "TALK BACK:",
    "Stock Index Futures Prices",
    "Calendar Of Canada Economic Data And Events",
    "NYMEX Opening",
    "Insci corp",
    "New York Interbank Foreign Exchange Rates",
    "Central & South West",
    "Qtr Financial Statements",
    "Shares In IPO",
    "London Closing Sugar",
    "London Late Afternoon Silver",
    "N Y Interbank Forex Rates",
    "Library of Congress",
    "Oslo Stock Prices",
    "New York Commodity Exchange",
    "New York Early Rates For Selected Currencies",
    "London Afternoon Gold Fixing",
    "New York Metal prices In U.S. Dollars",
    "Mexico News: ",
    "Cattle Prices",
    "French Unemployment",
    "Swedish Shrs",
    "Swedish Bonds",
    "Global Forex and Fixed Income Roundup",
    "Australian Morning Briefing",
    "Interbank Foreign Exchange Rates At",
    "Commodities Review:",
    "MARK TO MARKET: ",
    "DOW JONES GLOBAL INDEXES",
]


def filter_news(news_items_df):
    """
    This function filters the news items dataframe to remove news items that are not relevant to the
    analysis.
    
    Args:
      news_items_df: DataFrame
    
    Returns:
      A dataframe with the filtered news items.
    """

    news_items_filter_df = news_items_df.copy()
    news_items_filter_df = news_items_filter_df.loc[
        news_items_filter_df["product"].apply(
            lambda x: np.all([y not in x for y in filters_product])
        )
        & news_items_filter_df["subject"].apply(
            lambda x: np.all([y not in x for y in filters_subject])
        )
        & news_items_filter_df["stat"].apply(
            lambda x: np.all([y not in x for y in filters_stat])
        )
        & news_items_filter_df["company"].apply(
            lambda x: (len(x) == 0) or (len(x) == 1 and x[0] == "DJDAY")
        )
        & news_items_filter_df["journal"].apply(lambda x: len(x) == 0)
        & news_items_filter_df["headline"]
        .str.lower()
        .apply(lambda x: np.all([y.lower() not in x for y in filters_headline]))
    ]

    # Retail sales for other countries
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains(" Retail Sales")
            & ~news_items_filter_df["headline"].str.contains("US")
        )
    ]

    # Share prices/dividends
    # Something like: "Boston Private 2Q Net 9c A share Vs 6c >BPBC"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains("Q Net ")
            & news_items_filter_df["headline"].str.contains(" Vs ")
        )
    ]
    # Something like: "Analysts Expected 2Q Net 8c/Shr"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains("Q Net ")
            & news_items_filter_df["headline"].str.contains("/Shr")
        )
    ]
    # Something like: "Declares 5c Regular Quarterly Dividend"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains("Declares ")
            & news_items_filter_df["headline"].str.contains(
                "Regular Quarterly Dividend"
            )
        )
    ]

    # General obligations
    # Something like: "Montana $30M GOs -2: Bonds Due 2005-2016 Not Reoffered"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains("GOs")
            & news_items_filter_df["headline"].str.contains("$")
            & (
                news_items_filter_df["headline"].str.contains("Bonds")
                | news_items_filter_df["headline"].str.contains("Raised")
            )
        )
    ]

    # Results for certain companies
    # Like "Value Holdings Results -2-: 1Q Financial Table >VALH"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains("Results ")
            & news_items_filter_df["headline"].str.contains("Q Financial Table")
        )
    ]

    # Stock price changes for certain companies
    # Like "Culligan -2-: PWebber Sees FY96 Net At $1.11/Share >CUL"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains(" FY")
            & news_items_filter_df["headline"].str.contains("$")
            & news_items_filter_df["headline"].str.contains(">")
            & (
                news_items_filter_df["headline"].str.contains("/Sh")
                | news_items_filter_df["headline"].str.contains("Shs")
            )
        )
    ]

    # IPO Filing
    # Like "International Heritage Files 1.5M Sh IPO At $10/Sh  "
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains(" IPO")
            & (
                news_items_filter_df["headline"].str.contains("/Sh")
                | news_items_filter_df["headline"].str.contains("Shs")
            )
        )
    ]

    # Financial table
    # Like "Lance Inc. Earnings -3-: 6 Months Financial Table >LNCE    "
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains("Financial Table")
            & news_items_filter_df["headline"].str.contains("Earnings")
        )
    ]

    # Empty articles with just market direction
    # Just contain "(END) Dow Jones Newswires"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            (news_items_filter_df["text"].str.len() < 75)
            & (
                news_items_filter_df["headline"].str.contains("S&P 500")
                | news_items_filter_df["headline"].str.contains("DJIA")
            )
            & news_items_filter_df["headline"].str.contains("Points")
        )
    ]

    # Trading halts on specific stocks
    # Like: "Wheeler Real Estate Investment... Series D (WHLRD) Resumed Trading"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            (news_items_filter_df["text"].str.len() < 75)
            & (
                news_items_filter_df["headline"].str.contains("Resumed Trading")
                | news_items_filter_df["headline"].str.contains(
                    "Paused due to volatility"
                )
            )
            & news_items_filter_df["subject"].apply(lambda x: "N/STR" in x)
        )
    ]

    # Chapter 11 for specific stock
    # Like: "XiTec Inc. Files For Chapter 11 Bankruptcy Protection >XTIC"
    news_items_filter_df = news_items_filter_df.loc[
        ~(
            news_items_filter_df["headline"].str.contains("Chapter 11 Bankruptcy")
            & news_items_filter_df["headline"].str.contains(" >")
        )
    ]

    # About specific stock
    # Dreco Interview -2-: Sees Growth In Coiled Tubing >DREAF
    def check_stock_marker(headline):

        drop = False
        headline_split_space = headline.split(" ")
        if len(headline_split_space) >= 2:
            last_term = headline_split_space[-1]
            if len(last_term) > 1:
                drop = last_term[0] == ">"
        return drop

    news_items_filter_df = news_items_filter_df.loc[
        ~(news_items_filter_df["headline"].apply(lambda x: check_stock_marker(x)))
    ]
    # # Change in price targets
    # # Like: "Trust Price Target Cut to C$13.75/Share From C$14.25 by National Bank"
    # news_items_filter_df = news_items_filter_df.loc[
    #     (
    #         news_items_filter_df["headline"].str.contains("Price")
    #         # & (
    #         #     news_items_filter_df["headline"].str.contains("/Sh")
    #         #     | news_items_filter_df["headline"].str.contains("Shs")
    #         # )
    #     )
    # ]

    return news_items_filter_df


### Automatic News Detection
# The following functions are used to isolate relevant news
# in an automatic way


def extract_highlights_v1(highlights):

    highlights_split = highlights.split("\n\n")

    story_list = []
    story = []
    headline = None
    for idx, line in enumerate(highlights_split):

        # Generally the first line
        if len(line) == 0:
            continue

        # Check if its a headline (ends with '\n' or >50% uppercase)
        frac_string_upper = sum(1 for c in line if c.isupper()) / (len(line) + 1)
        if (
            (idx % 2 == 1)
            or (line.replace(" ", "")[-1:] == "\n")
            or (frac_string_upper > 0.5)
        ):

            # Wrap up previous story
            if headline:
                story_list.append([headline, "\n".join(story)])
                story = []

            headline = line.replace("\n", "").strip()

        if "(END) Dow Jones" in line:
            if headline:
                story_list.append([headline, "\n".join(story)])
            break

    story_df = pd.DataFrame(story_list, columns=["headline", "text"])

    # Clean up headlines
    story_df["headline"] = story_df["headline"].str.replace(
        "DOW JONES NEWSWIRES ANALYSIS AND COMMENTARY", ""
    )
    story_df["headline"] = story_df["headline"].str.replace("=", "")
    story_df["headline"] = story_df["headline"].str.lower()

    return story_df


def extract_highlights_v2(highlights, tag="top of the hour"):

    highlights_split = highlights.split("\n")

    headline_list = []
    start = False

    for line in highlights_split:

        # After we have passed the "Top of the Hour"
        # tag, start reading in headlines
        if start:
            # Empty highlight indicates that we are done
            if not len(line.strip()):
                if tag == "top of the day":
                    break

            # When there are gaps after top of the hour
            # tag need to ignore and use top of the
            # day tag
            if "top of the day" in line.lower():
                break

            # Otherwise
            headline_list.append(line)

        # If we see "Top of the Hour" tag, begin
        # reading
        else:
            if tag in line.lower():
                # Need to split by ">" in this case
                if len(line) > 50:
                    return extract_highlights_v3(highlights, tag=tag, retry=False)

                start = True

    # Handle empty headlines
    headline_list = [x for x in headline_list if len(x) > 10]

    story_df = pd.DataFrame(headline_list, columns=["headline"])

    # Case where there is no tag
    if not len(story_df):
        return extract_highlights_v4(highlights)

    return story_df


def extract_highlights_v3(highlights, tag="top of the day", retry=True):

    highlights_split = highlights.split("\n")

    headline_list = []

    for line in highlights_split:

        # If there is a line with just top of the whatever
        # we are in the v2 case
        if tag.lower() == line.lower().strip():
            return extract_highlights_v2(highlights, tag=tag)

        # DJNS style news highlights
        # Marked by "Top of the Day"
        if tag in line.lower():
            headline_list = line.lower().split(tag)[-1].split(">")

    headline_list = [x for x in headline_list if len(x) > 10]
    headline_list = [" ".join(x.split(" ")[1:]) for x in headline_list]

    story_df = pd.DataFrame(headline_list, columns=["headline"])

    if (not len(story_df)) and retry:
        return extract_highlights_v2(highlights, tag=tag)

    return story_df


def extract_highlights_v4(highlights):
    # This handles the case where a simple list of headlines
    # is given instead of having "Top of the ##" headings

    highlights_split = highlights.split("\n")

    headline_list = []

    for line in highlights_split:

        # Skip this type of header
        if "following items" in line.lower():
            continue

        # Empty highlight indicates that we are done
        if "(END)" in line:
            break

        # Otherwise
        headline_list.append(line)

    # Should just get rid of empty headlines
    headline_list = [x for x in headline_list if len(x) > 5]

    story_df = pd.DataFrame(headline_list, columns=["headline"])

    return story_df


def get_highlighted_news_diff(news_in_window, news_highlights_window):
    """Gets important news using difference in news highlights"""

    # Get time gap between highlights
    highlights_gap = (
        news_highlights_window["datetime"].iloc[0]
        - news_highlights_window["datetime"].iloc[-1]
    ).seconds / 60 ** 2
    if highlights_gap < 5:
        tag = "top of the hour"
    else:
        tag = "top of the day"

    # Get old and new highlights formatted as dataframes
    if news_highlights_window["datetime"].iloc[0] < pd.to_datetime("03-26-2007"):
        if news_highlights_window["headline"].str.contains("DJNS").iloc[0]:
            # DJNS style highlights
            highlights_new = extract_highlights_v3(
                news_highlights_window["text"].iloc[0], tag=tag
            )
            highlights_old = extract_highlights_v3(
                news_highlights_window["text"].iloc[1], tag=tag
            )
        else:
            highlights_new = extract_highlights_v2(
                news_highlights_window["text"].iloc[0], tag=tag
            )
            highlights_old = extract_highlights_v2(
                news_highlights_window["text"].iloc[1], tag=tag
            )
    else:
        highlights_new = extract_highlights_v1(news_highlights_window["text"].iloc[0])
        highlights_old = extract_highlights_v1(news_highlights_window["text"].iloc[1])

    # Check if there's the specific issue where
    # highlights cannot be seperated
    if len(highlights_new.iloc[0, 0]) > 300:
        logging.warning("Issue seperating highlights")
        print(news_highlights_window)
        print(highlights_new)
        print("=" * 100)

    # Do a fuzzy left merge between (highlights_new, highlights_old)
    # print_news(news_highlights_window)
    # display(highlights_new)
    # display(highlights_old)
    highlights_old_headlines = highlights_old["headline"].tolist()
    highlights_new[["headline_match", "match_level"]] = (
        highlights_new["headline"]
        .apply(lambda x: process.extract(x, highlights_old_headlines, limit=1)[0])
        .apply(pd.Series)
    )

    # Headlines in the new highlights that do not match the old highlights
    # very well must be novel
    highlights_new_novel = highlights_new.query("match_level < 90")[["headline"]].copy()

    # Get headlines within window
    news_in_window_headlines = news_in_window["headline"].to_list()

    # Deal with Dow Jones suffixes to titles
    for i, hline in enumerate(news_in_window_headlines):
        if " DJ " in hline[:10]:
            news_in_window_headlines[i] = hline.split(":")[-1].replace(" DJ ", "")

        news_in_window_headlines[i] = (
            news_in_window_headlines[i].strip().lower().replace("u.s.", "us")
        )

    # Match and filter to headlines with close news article matches
    highlights_new_novel[["news_headline_match", "news_match_level"]] = (
        highlights_new_novel["headline"]
        .apply(
            lambda x: process.extract(
                x, news_in_window_headlines, scorer=fuzz.QRatio, limit=1
            )[0]
        )
        .apply(pd.Series)
    )
    highlights_new_novel = highlights_new_novel.query("news_match_level > 65")

    # No matches
    if not len(highlights_new_novel):
        return news_in_window.iloc[:0, :]

    # News in window that later appears as a highlight
    news_in_window_highlights = news_in_window.iloc[
        highlights_new_novel["news_headline_match"].apply(
            lambda x: news_in_window_headlines.index(x)
        ),
        :,
    ]

    return news_in_window_highlights


def subset_news_highlights(news_highlights_window, event_datetime):
    """
    Given a window of news highlights and a datetime, return the two
    surrounding, most recent news headlines.
    
    Args:
      news_highlights_window: A dataframe containing the news highlights for a given
      event_datetime: The datetime of the event.
    
    Returns:
      A dataframe with the news highlights for the event
    """

    # If there are multiple highlights articles in one window
    # just grab two

    assert len(news_highlights_window) >= 2

    if len(news_highlights_window) > 2:
        news_highlights_window = pd.concat(
            [
                news_highlights_window.loc[
                    news_highlights_window["datetime"] > (event_datetime)
                ].iloc[-1],
                news_highlights_window.loc[
                    news_highlights_window["datetime"]
                    < (event_datetime - pd.Timedelta("15min"))
                ].iloc[0],
            ],
            axis=1,
        ).T

    return news_highlights_window


filters_highlights_headline = [
    "Canada News Highlights",
    "REPEAT:",
    "REPEAT&",
    "REPEAT/",
    "Business News H",
    "CORRECTION:",
    "DJbxwmcl ",  # Possible read error?
    "Special News Highlights",
    "Cda News Highlights",
    "CORRECT:",
]


def get_news_highlights_window_v1(djn_ym_df, event_datetime, window="15d"):

    # Get news in window
    news_in_window = find_associated_news(djn_ym_df, event_datetime, window=window)

    # Extract highlights - get only one per 15-min to avoid duplicated highlight articles
    news_in_window = news_in_window.loc[
        news_in_window["headline"]
        .str.lower()
        .apply(
            lambda x: np.all([y.lower() not in x for y in filters_highlights_headline])
        )
    ]
    news_highlights_window = (
        news_in_window.loc[
            news_in_window["headline"]
            .str.lower()
            .str.contains("News Highlights".strip().lower())
        ]
        .groupby(pd.Grouper(key="datetime", freq="15min"))
        .first()
        .dropna()
        .reset_index()
        .sort_values(by="datetime", ascending=False)
    )

    # If there aren't at least two highlights try again with larger window
    if len(news_highlights_window) < 2:
        raise Exception("Not enough articles")

    # Make sure headlines start with "News Highlights"
    # Could start with "DJ News Highlights"
    headlines = news_highlights_window["headline"].copy()
    headlines = headlines.str.replace(" DJ ", "")
    headlines = headlines.str.replace(" DJNS ", "")
    headlines = headlines.str.replace("Dow Jones ", "")
    headlines = headlines.str.strip()
    if not np.all(headlines.str.find("News Highlights") < 2):
        print(window)
        print(np.sort(headlines))
        print(news_highlights_window["headline"])
        raise Exception("Check headlines")

    # Deal with more than two articles
    news_highlights_window = subset_news_highlights(
        news_highlights_window, event_datetime
    )

    # Make sure there are two articles: the old and new highlights
    assert len(news_highlights_window) == 2

    # Filter down news in window down to the same day
    news_in_window = (
        news_in_window.set_index("datetime")
        .loc[event_datetime.strftime("%Y-%m-%d")]
        .reset_index()
    )

    return news_in_window, news_highlights_window


def get_news_highlights_window_v2(djn_ym_df, event_datetime, window="5d"):

    # Get news in window
    news_in_window = find_associated_news(djn_ym_df, event_datetime, window=window)

    # Get news highlights
    news_highlights_window = news_in_window.loc[
        news_in_window["headline"]
        .str.lower()
        .str.contains("News Highlights: Top Econ".strip().lower())
    ].sort_values(by="datetime", ascending=False)

    # Make sure there are at least two articles: the old and new highlights
    if len(news_highlights_window) < 2:

        # Get news in larger window
        news_in_window = find_associated_news(djn_ym_df, event_datetime, window=window)

        # And use global highlights instead
        news_highlights_window = news_in_window.loc[
            news_in_window["headline"]
            .str.lower()
            .str.contains("News Highlights: Top Global".strip().lower())
        ].sort_values(by="datetime", ascending=False)

    # Deal with more than two articles
    news_highlights_window = subset_news_highlights(
        news_highlights_window, event_datetime
    )

    # Filter down news in window down to the same day
    news_in_window = (
        news_in_window.set_index("datetime")
        .loc[event_datetime.strftime("%Y-%m-%d")]
        .reset_index()
    )

    return news_in_window, news_highlights_window


def get_news_in_window_highlights(djn_ym_df, event_datetime):
    """
    Given an event, return adjacent news highlights.
    
    Args:
      djn_ym_df: DataFrame of news data for the year and month of the event
      event_datetime: datetime object for the event
    
    Returns:
      A list of news highlights around the event datetime.
    """

    if event_datetime < pd.to_datetime("03-26-2007"):
        # Get news highlights in window
        news_in_window, news_highlights_window = get_news_highlights_window_v1(
            djn_ym_df, event_datetime
        )
    else:
        # Will only work for March 26 2007 later?
        news_in_window, news_highlights_window = get_news_highlights_window_v2(
            djn_ym_df, event_datetime
        )

    # Extract new highlights
    news_in_window_highlights = get_highlighted_news_diff(
        news_in_window, news_highlights_window
    )

    return news_in_window_highlights


def print_news_on_datetime(news_datetime):
    """
    Prints news for a given datetime.
    
    Args:
      news_datetime: The datetime of the news event.
    
    Returns:
      None
    """

    if type(news_datetime) == str:
        news_datetime = pd.to_datetime(news_datetime)

    # Get data for some month
    ym = news_datetime.strftime("%Y-%m")
    djn_ym_df = pd.read_parquet(f"../../data/dowjones/preproc/{ym}.parquet")
    djn_ym_df["datetime"] = (
        djn_ym_df["display_date"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    )

    # Filter down to the event time
    djn_ym_filter_df = djn_ym_df.loc[
        djn_ym_df["datetime"].dt.strftime("%Y-%m-%d %H%M")
        == news_datetime.strftime("%Y-%m-%d %H%M")
    ]

    # Print articles
    print_news(djn_ym_filter_df)
