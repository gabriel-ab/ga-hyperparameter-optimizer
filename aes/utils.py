import numpy as np
from pandas import DataFrame


def random_split_df(df: DataFrame, split_ratio: float):
    data_size = len(df)
    split_size = int(data_size * split_ratio)
    indices = np.random.permutation(data_size)
    train_indices, test_indices = indices[:split_size], indices[split_size:]
    train_data, test_data = df.iloc[train_indices], df.iloc[test_indices]
    return train_data, test_data
