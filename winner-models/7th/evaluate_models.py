import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics

from evaluate import Evaluation
from models import LGBMHierarchModel
from preprocess import data_pipeline

model_map = {"lgbmhierarch": (LGBMHierarchModel, {})}


def generate_dataset(args):
    """
    Generates the dataset.
    Args:
        args (dict): args from argparser
    Returns:
        Data: Train/Test datasets
    """
    return data_pipeline(args["datapath"])


def evaluate_model(args, train, test):
    """
    trains the model given by the args argument on the corresponding data
    Args:
        args (dict): Args from arg parser
        data (pyg Data): Data to train on
    """
    np.random.seed(args["seed"])

    models = [m for m in args["models"].split(",")]
    eva_metrics = {
        "rmse": (metrics.mean_squared_error, {"squared": False}),
        "mae": (metrics.mean_absolute_error, {}),
        "mape": (metrics.mean_absolute_percentage_error, {}),
    }
    eva = Evaluation()

    for m in models:
        model, margs = model_map[m]
        model = model(device=None, seed=args["seed"], **margs)
        model.train(train.copy(), use_params=True)
        eva.register_model(m, model)

    path = os.path.join(
        args["path"],
        str(datetime.now().strftime("%m-%d-%Y-%H-%M")),
    )
    os.mkdir(path)

    eva.run(save_dir=path, metrics=eva_metrics, test=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Evaluation")
    parser.add_argument(
        "-m", "--models", help="Models to evaluate", required=True, type=str
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Save path evaluation results",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-dp",
        "--datapath",
        help="load path for dataset",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-se",
        "--seed",
        help="Seed for the random operations like train/test split",
        default=69,
        type=int,
    )

    args = vars(parser.parse_args())

    train, test = generate_dataset(args)
    evaluate_model(args, train, test)
