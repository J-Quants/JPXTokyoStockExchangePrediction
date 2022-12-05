# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os

import numpy as np  # linear algebra
import optuna
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


for dirname, _, filenames in os.walk("input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
stock_prices = pd.read_csv(
    "input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv"
)

# %%
stock_list = pd.read_csv("input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
target_stock_list = stock_list[stock_list["Universe0"]]
sec_info = target_stock_list[["SecuritiesCode", "33SectorName", "17SectorName"]]
stock_prices = pd.merge(stock_prices, sec_info, on="SecuritiesCode")
stock_prices["33SectorName"] = stock_prices["33SectorName"].astype("category")
stock_prices["17SectorName"] = stock_prices["17SectorName"].astype("category")

# %%
# Forwardfill NaN-values
print(stock_prices["Target"].isnull().values.any())
stock_prices.update(stock_prices.groupby("SecuritiesCode")["Target"].ffill().fillna(0))
print(stock_prices["Target"].isnull().values.any())
print(stock_prices.columns)

# %%
def get_model_data(df, cols):
    df_stock = df.copy()

    df_stock = df_stock.sort_values(by="Date", ascending=False).reset_index(drop=True)
    df_stock["average"] = (df_stock["High"] + df_stock["Low"] + df_stock["Close"]) / 3
    df_stock["Date"] = pd.to_datetime(df_stock["Date"], format="%Y-%m-%d")
    df_stock["dayofweek"] = df_stock["Date"].dt.dayofweek
    df_stock["is_quater_start"] = df_stock["Date"].dt.is_quarter_start.map(
        {False: 0, True: 1}
    )
    df_stock["is_month_start"] = df_stock["Date"].dt.is_month_start.map(
        {False: 0, True: 1}
    )
    df_stock["is_month_end"] = df_stock["Date"].dt.is_month_end.map({False: 0, True: 1})
    # Another feature day of the week will also be added.
    df_stock = df_stock.sort_values(by="Date").reset_index(drop=True)
    df_model = df_stock[cols]
    return df_model


# %%
cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "average",
    "dayofweek",
    "is_quater_start",
    "is_month_start",
    "is_month_end",
    "Target",
    "RowId",
    "Date",
    "SecuritiesCode",
    "Volume",
]
stock_prices = get_model_data(stock_prices, cols)

# %%
stock_prices = stock_prices.infer_objects()
stock_prices.head(5)

# %%
# Form 31 Targetgroups
stock_prices["Target"] = (
    stock_prices.groupby("Date")["Target"]
    .rank("dense", ascending=False)
    .astype(int, errors="ignore")
)
stock_prices["Target"] = pd.qcut(stock_prices.Target, 30).cat.codes
stock_prices.head()

# %%
# Just some arbitrary dates
time_config = {"train_split_date": "2020-12-23"}

train = stock_prices[stock_prices.Date >= time_config["train_split_date"]]

print(train.shape)
# print(test.shape)

col_use = [c for c in stock_prices.columns if c not in ["RowId", "Date", "Target"]]

# %%
"""Implements model selection methods."""

from itertools import groupby
from typing import Iterable, Iterator, Optional, Tuple, cast

import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class GroupTimeSeriesSplit:
    """Time series cross validation with custom grouping."""

    def __init__(
        self,
        test_size: int,
        train_size: Optional[int] = None,
        n_splits: Optional[int] = None,
        gap: int = 0,
        shift_size: int = 1,
        window="rolling",
    ) -> None:
        """Initializes cross validation parameters.
        Args:
            test_size (int):
                Size of test dataset.
            train_size (Optional[int], optional):
                Size of train dataset. Defaults to None.
            n_splits (int, optional):
                Number of splits. Defaults to None.
            gap (int, optional):
                Gap size. Defaults to 0.
            shift_size (int, optional):
                Step to shift for the next fold. Defaults to 1.
            window (str):
                Type of the window. Defaults to 'rolling'.
        """
        if (train_size is None) and (n_splits is None):
            raise ValueError("Either train_size or n_splits have to be defined")

        if window not in ["rolling", "expanding"]:
            raise ValueError('Window can be either "rolling" or "expanding"')

        if (train_size is not None) and (window == "expanding"):
            raise ValueError("Train size can be specified only with rolling window")

        self.test_size = test_size
        self.train_size = train_size
        self.n_splits = n_splits
        self.gap = gap
        self.shift_size = shift_size
        self.window = window

    def split(
        self,
        X: Iterable,
        y: Optional[Iterable] = None,
        groups: Optional[Iterable] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Calculates train/test indices based on split parameters.
        Args:
            X (Iterable):
                Dataset with features.
            y (Iterable):
                Dataset with target.
            groups (Iterable):
                Array with group numbers.
        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]:
                Train/test dataset indices.
        """
        test_size = self.test_size
        gap = self.gap
        shift_size = self.shift_size

        # Convert to indexable data structures with additional lengths consistency check
        X, y, groups = indexable(X, y, groups)

        # Check if groups are specified
        if groups is None:
            raise ValueError("Groups must be specified")

        # Check if groups are sorted in dataset
        group_seqs = [group[0] for group in groupby(groups)]
        unique_groups, group_starts_idx = np.unique(groups, return_index=True)
        n_groups = _num_samples(unique_groups)
        self._n_groups = n_groups

        if group_seqs != sorted(unique_groups):
            raise ValueError("Groups must be presorted in increasing order")

        # Create mapping between groups and its start indices in array
        groups_dict = dict(zip(unique_groups, group_starts_idx))

        # Calculate number of samples
        n_samples = _num_samples(X)

        # Calculate remaining split params
        self._calculate_split_params()
        train_size = cast(int, self.train_size)
        n_splits = cast(int, self.n_splits)
        train_start_idx = self._train_start_idx

        # Calculate start/end indices for initial train/test datasets
        train_end_idx = train_start_idx + train_size
        test_start_idx = train_end_idx + gap
        test_end_idx = test_start_idx + test_size

        # Process each split
        for _ in range(n_splits):
            # Calculate train indices range
            train_idx = np.r_[
                slice(
                    groups_dict[group_seqs[train_start_idx]],
                    groups_dict[group_seqs[train_end_idx]],
                )
            ]

            # Calculate test indices range
            if test_end_idx < n_groups:
                test_idx = np.r_[
                    slice(
                        groups_dict[group_seqs[test_start_idx]],
                        groups_dict[group_seqs[test_end_idx]],
                    )
                ]
            else:
                test_idx = np.r_[
                    slice(groups_dict[group_seqs[test_start_idx]], n_samples)
                ]

            # Yield train/test indices range
            yield (train_idx, test_idx)

            # Shift train dataset start index by shift size for rolling window
            if self.window == "rolling":
                train_start_idx = train_start_idx + shift_size

            # Shift train dataset end index by shift size
            train_end_idx = train_end_idx + shift_size

            # Shift test dataset indices range by shift size
            test_start_idx = test_start_idx + shift_size
            test_end_idx = test_end_idx + shift_size

    def get_n_splits(
        self,
        X: Iterable,
        y: Optional[Iterable] = None,
        groups: Optional[Iterable] = None,
    ) -> int:
        """Calculates number of splits given specified parameters.
        Args:
            X (Iterable):
                Dataset with features. Defaults to None.
            y (Optional[Iterable], optional):
                Dataset with target. Defaults to None.
            groups (Optional[Iterable], optional):
                Array with group numbers. Defaults to None.
        Returns:
            int:
                Calculated number of splits.
        """
        if self.n_splits is not None:
            return self.n_splits
        else:
            raise ValueError("Number of splits is not defined")

    def _calculate_split_params(self) -> None:
        train_size = self.train_size
        test_size = self.test_size
        n_splits = self.n_splits
        gap = self.gap
        shift_size = self.shift_size
        n_groups = self._n_groups

        not_enough_data_error = (
            "Not enough data to split number of groups ({0})"
            " for number splits ({1})"
            " with train size ({2}),"
            " test size ({3}), gap ({4}), shift_size ({5})"
        )

        if (train_size is None) and (n_splits is not None):
            train_size = n_groups - gap - test_size - (n_splits - 1) * shift_size
            self.train_size = train_size

            if train_size <= 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups, n_splits, train_size, test_size, gap, shift_size
                    )
                )
            train_start_idx = 0
        elif (n_splits is None) and (train_size is not None):
            n_splits = (n_groups - train_size - gap - test_size) // shift_size + 1
            self.n_splits = n_splits

            if self.n_splits <= 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups, n_splits, train_size, test_size, gap, shift_size
                    )
                )
            train_start_idx = (
                n_groups - train_size - gap - test_size - (n_splits - 1) * shift_size
            )
        elif (n_splits is not None) and (train_size is not None):
            train_start_idx = (
                n_groups - train_size - gap - test_size - (n_splits - 1) * shift_size
            )

            if train_start_idx < 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups, n_splits, train_size, test_size, gap, shift_size
                    )
                )

        self._train_start_idx = train_start_idx


# %%
from lightgbm import LGBMRanker
from sklearn.metrics import mean_squared_error, ndcg_score
from sklearn.preprocessing import LabelEncoder


def objective(trial, train=train, col_use=col_use):
    param = {
        "boosting_type": "dart",
        "n_jobs": 3,
        "num_iterations": 500,
        "num_leaves": trial.suggest_int("num leaves", low=10, high=100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "subsample": trial.suggest_categorical(
            "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
        ),
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [100, 400, 700, 1000]
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", [-1, 5, 7, 9, 11, 13, 15, 17]
        ),
        "random_state": trial.suggest_categorical("random_state", [2020]),
        "min_child_weight": trial.suggest_int("min_child_weight", low=1e-4, high=1),
    }

    # time series cross val here
    avg_score = 0
    le = LabelEncoder()
    train["groups"] = le.fit_transform(train["Date"])
    train.sort_values("groups", inplace=True)
    for train_idx, test_idx in GroupTimeSeriesSplit(test_size=10, n_splits=5).split(
        train, groups=train["groups"]
    ):
        qtrain = [train.iloc[train_idx].shape[0] / 2000] * 2000
        qval = [train.iloc[test_idx].shape[0] / 2000] * 2000
        model = LGBMRanker(**param)
        model.fit(
            train.iloc[train_idx][col_use],
            train.iloc[train_idx]["Target"],
            group=qtrain,
            eval_set=[(train.iloc[test_idx][col_use], train.iloc[test_idx]["Target"])],
            eval_group=[qval],
            eval_at=[1],
            verbose=100,
        )
        score = model.evals_result_["valid_0"]["ndcg@1"][-1]

        avg_score += score

    return avg_score


# %%
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=3)

best_trial = study.best_trial.params
print(best_trial)
