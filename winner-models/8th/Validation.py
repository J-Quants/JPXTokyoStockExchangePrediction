import pandas as pd


def KFoldDataPartition(df, K=5):
    df["Date"] = pd.to_datetime(df["Date"])
    dates = sorted(df["Date"].unique())  # sort just in case

    datasets = []
    indices = [int(i / K * len(dates)) for i in range(K + 1)]
    indices[-1] -= 1

    for train_start in range(len(indices[:-1])):
        for train_end in range(train_start + 1, len(indices)-1):
            start_date = dates[indices[train_start]]
            end_date = dates[indices[train_end]]
            val_end_date = dates[indices[train_end + 1]]
            df_train = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
            df_val = df[(df["Date"] > end_date) & (df["Date"] <= val_end_date)]
            datasets.append((df_train, df_val))

    return datasets