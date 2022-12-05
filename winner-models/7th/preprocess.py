import os
from decimal import ROUND_HALF_UP, Decimal
from typing import Tuple

import numpy as np
import pandas as pd

""" 
Methods for preprocessing the dataset 
"""


def data_pipeline(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and merge datasets

    Args:
        dir_path (str): path to data dir

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train/Test set
    """

    train = pd.read_csv(os.path.join(dir_path, "train_files/stock_prices.csv"))
    test = pd.read_csv(os.path.join(dir_path, "supplemental_files/stock_prices.csv"))
    stock_list = pd.read_csv(os.path.join(dir_path, "stock_list.csv"))
    target_stock_list = stock_list[stock_list["Universe0"]]

    train = train.drop(["ExpectedDividend", "RowId"], axis=1)
    # train = adjust_price(train)
    train = train.fillna(0)
    test = test.drop(["ExpectedDividend", "RowId"], axis=1)
    # test = adjust_price(test)
    test = test.fillna(0)

    # merge stock categories as additional features
    sec_info = target_stock_list[["SecuritiesCode", "33SectorName", "17SectorName"]]
    train = pd.merge(train, sec_info, on="SecuritiesCode")
    train["33SectorName"] = train["33SectorName"].astype("category")
    train["17SectorName"] = train["17SectorName"].astype("category")

    # use supplemental stock prices as test set to evaluate performance of classifiers
    test = pd.merge(test, sec_info, on="SecuritiesCode")
    test["33SectorName"] = test["33SectorName"].astype("category")
    test["17SectorName"] = test["17SectorName"].astype("category")

    train.update(train.groupby("SecuritiesCode")["Target"].ffill().fillna(0))
    test.update(test.groupby("SecuritiesCode")["Target"].ffill().fillna(0))

    # # add features
    # train = add_time_features(train)
    # test = add_time_features(test)

    train["SupervisionFlag"] = train["SupervisionFlag"].map({True: 1, False: 0})
    test["SupervisionFlag"] = test["SupervisionFlag"].map({True: 1, False: 0})

    # cut timeframe where not all targets are present
    time_config = {"train_split_date": "2020-12-23"}
    train = train[train.Date >= time_config["train_split_date"]]

    return train, test


def add_time_features(df: pd.DataFrame):
    df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)
    df["average"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_quarter_start"] = df["Date"].dt.is_quarter_start.map({False: 0, True: 1})
    df["is_month_start"] = df["Date"].dt.is_month_start.map({False: 0, True: 1})
    df["is_month_end"] = df["Date"].dt.is_month_end.map({False: 0, True: 1})
    # Another feature day of the week will also be added.
    df = df.sort_values(by="Date").reset_index(drop=True)

    return df


def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(
            lambda x: float(
                Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            )
        )
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = (
        price.groupby("SecuritiesCode")
        .apply(generate_adjusted_close)
        .reset_index(drop=True)
    )
    return price
