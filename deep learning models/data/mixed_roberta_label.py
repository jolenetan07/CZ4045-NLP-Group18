import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot


def onehot(x, c):
    return np.eye(c, dtype=int)[x.reshape(-1)]


def get_dataloaders(batch_size, data_path='dataset/biden_tweets_manually_labeled_roberta_labeled.csv'):
    df = pd.read_csv(data_path)

    tokenized_col_name = "Tokenized"

    roberta_full_series = df['roberta_labeled']
    polarity_full_series = df['polarity']
    tokens_full_series = df[tokenized_col_name]

    roberta_full_nparr = np.asarray(a=roberta_full_series.to_list(), dtype=int)
    polarity_full_nparr = np.asarray(
        a=polarity_full_series.to_list(), dtype=int)

    roberta_onehot = onehot(roberta_full_nparr, 3)
    tokens_full_nparr = np.asarray(a=[list_string_to_np_arr(
        x) for x in tokens_full_series.to_list()], dtype=int)
    concat_input = np.concatenate((tokens_full_nparr, roberta_onehot), axis=1,)

    train_valid_split_point = int(len(tokens_full_nparr) * 0.8)
    valid_test_split_point = int(len(tokens_full_nparr) * 0.9)

    train_tokens = concat_input[: train_valid_split_point]
    valid_tokens = concat_input[train_valid_split_point: valid_test_split_point]
    test_tokens = concat_input[valid_test_split_point:]

    train_polarity = polarity_full_nparr[: train_valid_split_point]
    valid_polarity = polarity_full_nparr[train_valid_split_point: valid_test_split_point]
    test_polarity = polarity_full_nparr[valid_test_split_point:]

    train_data = TensorDataset(torch.from_numpy(
        train_tokens), torch.from_numpy(train_polarity))
    valid_data = TensorDataset(torch.from_numpy(
        valid_tokens), torch.from_numpy(valid_polarity))
    test_data = TensorDataset(torch.from_numpy(
        test_tokens), torch.from_numpy(test_polarity))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


def list_string_to_np_arr(string):
    string = string.replace("[", "")
    string = string.replace("]", "")
    str_list = string.split(",")
    return np.asarray(str_list, dtype=int)




