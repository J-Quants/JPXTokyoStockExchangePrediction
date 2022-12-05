import numpy as np
import pandas as pd
from enum import Enum


class FeatureType(Enum):
    GLOBAL = 0
    LOCAL = 1


class Feature:

    def __init__(self, feature_type, name):
        self.feature_type = feature_type
        self.name = name

    def add_feature_pandas(self, df):
        raise NotImplementedError

    def update_row(self, row):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class SMA(Feature):

    def __init__(self, col, period):
        super().__init__(FeatureType.LOCAL, col + "SMA" + str(period))
        self.col = col
        self.period = period
        self.elements = [np.nan] * period
        self.ptr = 0
        self.mean = np.nan

    def add_feature_pandas(self, df):
        df[self.name] = df[self.col].rolling(self.period).mean()
        for k in range(self.period):
            df[self.name].iloc[k] = np.mean(df[self.col].iloc[:k+1])
        return df

    def update_row(self, row):

        dequeue = self.elements[self.ptr % self.period]
        enqueue = row[self.col]
        self.elements[self.ptr % self.period] = enqueue
        self.ptr += 1
        mean = 0
        if self.ptr < self.period:  # We have not yet seen enough elements
            mean = np.mean(self.elements[:self.ptr])
        elif self.ptr == self.period:  # We have the value for the first time
            self.mean = np.mean(self.elements)
            mean = self.mean
        else:
            mean = self.mean + (- dequeue + enqueue) / self.period  # Simple and efficient updates
            self.mean = mean

        row[self.name] = mean
        return row

    def reset(self):
        self.elements = [np.nan] * self.period
        self.ptr = 0
        self.mean = np.nan

    def copy(self):
        return SMA(self.col, self.period)


class Amplitude(Feature):
    def __init__(self):
        super().__init__(FeatureType.LOCAL, "Amplitude")

    def add_feature_pandas(self, df):
        df[self.name] = df["High"] - df["Low"]
        return df

    def update_row(self, row):
        row[self.name] = row["High"] - row["Low"]
        return row

    def reset(self):
        return

    def copy(self):
        return Amplitude()


class OpenCloseReturn(Feature):
    def __init__(self):
        super().__init__(FeatureType.LOCAL, "OpenCloseReturn")

    def add_feature_pandas(self, df):
        df[self.name] = (df["Close"] - df["Open"]) / df["Open"]
        return df

    def update_row(self, row):
        row[self.name] = (row["Close"] - row["Open"]) / row["Open"]
        return row

    def reset(self):
        return

    def copy(self):
        return OpenCloseReturn()


class Return(Feature):
    def __init__(self):
        super().__init__(FeatureType.LOCAL, "Return")
        self.last_close = None

    def add_feature_pandas(self, df):
        df[self.name] = ((df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)).fillna(0)
        return df

    def update_row(self, row):
        if self.last_close is None:
            row[self.name] = 0
        else:
            row[self.name] = (row["Close"] - self.last_close) / self.last_close
        self.last_close = row["Close"]
        return row

    def reset(self):
        self.last_close = None

    def copy(self):
        return Return()


class Volatility(Feature):

    def __init__(self, n=30):
        super().__init__(FeatureType.LOCAL, "Volatility" + str(n))
        self.n = n
        self.returns = np.ones(n) * np.nan
        self.ptr = 0
        self.index = 0

    def volatility_row_function(self, df, row):
        l = max(0, self.index + 1 - self.n)
        r = self.index + 1
        self.index += 1
        return np.std(df["Return"].to_numpy()[l:r], ddof=1)

    def add_feature_pandas(self, df):
        df[self.name] = df.apply(lambda row: self.volatility_row_function(df, row), axis=1)
        return df.fillna(0)

    def update_row(self, row):
        self.returns[self.ptr % self.n] = row["Return"]

        if self.ptr == 0:
            row[self.name] = 0
        elif self.ptr < self.n - 1:
            vec = self.returns[:(self.ptr + 1) % self.n]
            row[self.name] = np.std(vec, ddof=1)
        else:
            row[self.name] = np.std(self.returns, ddof=1)

        self.ptr += 1
        return row

    def reset(self):
        self.returns = np.ones(self.n) * np.nan
        self.ptr = 0
        self.index = 0

    def copy(self):
        return Volatility(self.n)


class FeatureChecker:
    def verify(feature, df_prices):
        df_pandas = feature.add_feature_pandas(df_prices)
        df_online = df_prices.apply(lambda row: feature.update_row(row), axis=1)

        return np.isclose(df_pandas[feature.name].to_numpy(),
                          df_online[feature.name].to_numpy(), equal_nan=True).all()

    def verify_features(features, df_prices):
        verifications = []
        all_verified = True
        for feature in features:
            this = FeatureChecker.verify(feature, df_prices)
            verifications.append((feature.name, this))
            all_verified = all_verified and this

        if all_verified:
            print("All features passed the check.")
        else:
            print("Some features failed the check.")

        print(verifications)

        return verifications