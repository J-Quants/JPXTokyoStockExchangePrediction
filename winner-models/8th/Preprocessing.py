import numpy as np
import pandas as pd


class StockDataPreprocessor:

    def fill_nans(df_stocks):
        # Dividends NaNs means 0 zero.
        df_stocks["ExpectedDividend"] = df_stocks["ExpectedDividend"].fillna(0)
        subdfs = []
        for stock_id, subdf in df_stocks.groupby("SecuritiesCode"):
            for i in range(len(subdf)):
                if not np.isnan(subdf.iloc[i]["Open"]):
                    break
            subdf = subdf.iloc[i:]
            subdf = subdf.fillna(method="ffill")
            subdfs.append(subdf)

            if i != 0:
                print(f"Stock id {stock_id} dropping {i} rows")

        new_df_stocks = pd.concat(subdfs).sort_index().reset_index(drop=True)

        print(f"Number of rows dropped is {len(df_stocks) - len(new_df_stocks)}")
        return new_df_stocks

    def add_cum_adj_factor(df_stocks):
        cum_adj_list = []
        for stock_id, subdf in df_stocks.groupby("SecuritiesCode"):
            cum_adj = subdf["AdjustmentFactor"].cumprod().shift(1, fill_value=1)
            cum_adj_list.append(cum_adj)

        df_stocks["CumAdjFactor"] = pd.concat(cum_adj_list).sort_index().to_numpy()
        return df_stocks

    def adjust_prices_and_volume(df_stocks):
        df_stocks["Open"] = df_stocks["Open"] / df_stocks["CumAdjFactor"]
        df_stocks["High"] = df_stocks["High"] / df_stocks["CumAdjFactor"]
        df_stocks["Low"] = df_stocks["Low"] / df_stocks["CumAdjFactor"]
        df_stocks["Close"] = df_stocks["Close"] / df_stocks["CumAdjFactor"]
        df_stocks["Volume"] = df_stocks["Volume"] * df_stocks["CumAdjFactor"]
        return df_stocks

    def preprocess_for_training(df_stocks):
        df_stocks = StockDataPreprocessor.fill_nans(df_stocks)
        df_stocks = StockDataPreprocessor.add_cum_adj_factor(df_stocks)
        df_stocks = StockDataPreprocessor.adjust_prices_and_volume(df_stocks)
        return df_stocks