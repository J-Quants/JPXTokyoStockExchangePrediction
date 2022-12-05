import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.preprocessing import LabelEncoder
from utils import GroupTimeSeriesSplit, calc_spread_return_sharpe

from .model import Model


class LGBMHierarchModel(Model):
    def __init__(self, device=None, seed=69):
        self.seed = seed
        self._best_found_params = {
            "num_leaves": 17,
            "learning_rate": 0.014,
            "n_estimators": 700,
            "max_depth": -1,
        }
        self.models = {}

    def train(self, train: pd.DataFrame, use_params=False):
        if not use_params:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self._tune(trial, X, y), n_trials=5)
            best_trial = study.best_trial.params
        else:
            best_trial = self._best_found_params

        for name, group in train.groupby("33SectorName"):
            y = group["Target"].to_numpy()
            X = group.drop(["Target"], axis=1)
            X = X.drop(["Date", "SecuritiesCode"], axis=1)
            model = LGBMRegressor(**best_trial)
            model.fit(X, y, verbose=False)
            self.models[name] = model

    def predict(self, test: pd.DataFrame, metrics):
        res, count = {}, 0
        for name, group in test.groupby("33SectorName"):
            y_test = group["Target"].to_numpy()
            X_test = group.drop(["Target", "Date", "SecuritiesCode"], axis=1)

            y_pred = self.models[name].predict(X_test)
            for name, (func, args) in metrics.items():
                if name not in res:
                    res[name] = func(y_test, y_pred, **args)
                else:
                    res[name] += func(y_test, y_pred, **args)
            count += 1

        for k, v in res.items():
            res[k] = v / count
        return res

    def _tune(self, trial, X, y):
        param = {
            "boosting_type": "dart",
            "n_jobs": -1,
            "num_iterations": 500,
            "num_leaves": trial.suggest_int("num leaves", low=10, high=100),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
            ),
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [100, 400, 700, 1000]
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [-1, 5, 7, 9, 11, 13, 15, 17]
            ),
            "metric": "mae",
            "random_state": self.seed,
        }

        avg_score = 0
        splits = 5
        le = LabelEncoder()
        X["groups"] = le.fit_transform(X["Date"])
        X.sort_values("groups", inplace=True)

        for train_idx, test_idx in GroupTimeSeriesSplit(
            test_size=90, n_splits=splits
        ).split(X, groups=X["groups"]):
            X_train, y_train = X.iloc[train_idx, :], y[train_idx]
            X_val, y_val = X.iloc[test_idx, :], y[test_idx]
            X_train.drop(["Date", "SecuritiesCode"], axis=1, inplace=True)
            X_val = X_val[
                X_val.columns[~X_val.columns.isin(["Date", "SecuritiesCode"])]
            ]

            model = LGBMRegressor(**param)
            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (
                        X_val,
                        y_val,
                    )
                ],
                verbose=100,
            )

            y_pred = model.predict(X_val)
            avg_score += mean_squared_error(y_val, y_pred, squared=False)

        return avg_score / splits
