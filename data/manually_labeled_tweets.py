import string

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import re

import nltk

overall_tokens = []

def get_dataloaders(batch_size, data_path="dataset/biden_tweets_labeled.csv"):
    """
    Around 1000 data records, that is manually labeled
    """
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("vader_lexicon")
    """
    things to download on the fly if not using py ide:
    stopwords
    punkt
    vader_lexicon
    
    """

    # nltk.download()


    text_col_name = "Text"
    subjectivity_label_name = "subjectivity"
    polarity_label_name = "polarity"
    token_col_name = "Raw tokens"
    tokenized_col_name = "Tokenized"
    length_col_name = "Token length"
    ref_sentiment_name = "NLTK ref sentiment"

    columns_to_read = [text_col_name, subjectivity_label_name, polarity_label_name]



    use_csv_col_as_idx = False

    # tweets_csv = pd.read_csv(data_path)
    # if tweets_csv.columns[0] == "Unnamed: 0":
    if use_csv_col_as_idx:
        print(f"first column as index, reading csv")
        tweets_csv = pd.read_csv(data_path, index_col=[0])
    else:
        print(f"first column is named, fall back to specify used_cols")
        tweets_csv = pd.read_csv(data_path, usecols=columns_to_read)

    stopwords = nltk.corpus.stopwords.words("english")



    """
    punct to replace: 
    ’ to '
    ` to '
    
    
    """

    def remove_at_tags(x: pd.Series):
        x[text_col_name]: str
        words = x[text_col_name].split()
        for idx in range(len(words)):
            words[idx] = words[idx].replace("’", "'")
            words[idx] = words[idx].replace("`", "'")

        words = [x if not re.match(r"https?:", x) else "website_name" for x in words]
        words_w_at_tags = [x for x in words if not re.match(r".*@.*", x)]

        result = ''
        for elem in words_w_at_tags:
            result += elem + ' '
        return result

    def tweet_en_tokenize(x: pd.Series):
        global overall_tokens
        tokens = word_tokenize(x[text_col_name])
        tokens_w_stops = [x for x in tokens if x not in stopwords]
        overall_tokens += tokens_w_stops
        return tokens_w_stops

    label_map_dict = {2: 0.5, 1: 1, 0: 0}

    def apply_self_mapping_of_label(x: pd.Series):
        return label_map_dict[x[polarity_label_name]]

    tweets_csv[text_col_name] = tweets_csv.apply(remove_at_tags, axis=1)

    tweets_csv[token_col_name] = tweets_csv.apply(tweet_en_tokenize, axis=1)

    tweets_csv[polarity_label_name] = tweets_csv.apply(apply_self_mapping_of_label, axis=1)

    tweets_csv[length_col_name] = tweets_csv.apply(lambda x: len(x[token_col_name]), axis=1)

    tweets_csv = tweets_csv[tweets_csv[length_col_name] <= 50]

    tweet_freq_dict = nltk.FreqDist(overall_tokens)
    print(type(tweet_freq_dict))
    tweet_freq_dict.tabulate(25)

    vocab_to_int_encoding = {pair[1]: pair[0] + 1 for pair in enumerate(tweet_freq_dict)}
    print(len(vocab_to_int_encoding))
    print(type(vocab_to_int_encoding))

    truncate_to_unknown_corpus_length_limit = 5000
    assert truncate_to_unknown_corpus_length_limit < len(
        vocab_to_int_encoding), "unknown truncation limit must be smaller than corpus length"
    vocab_to_int_encoding["<unk>"] = truncate_to_unknown_corpus_length_limit + 1

    def tokens_to_int(x: pd.Series):
        tokens = x[token_col_name]
        try:
            tokens_in_int = [vocab_to_int_encoding[token] for token in tokens]
            for idx in range(len(tokens_in_int)):
                if tokens_in_int[idx] >= truncate_to_unknown_corpus_length_limit:
                    tokens_in_int[idx] = truncate_to_unknown_corpus_length_limit + 1
        except KeyError:
            print(x)
            return -1
        return tokens_in_int

    tweets_csv[tokenized_col_name] = tweets_csv.apply(tokens_to_int, axis=1)

    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(overall_tokens)
    bigram_finder.ngram_fd.tabulate(10)

    trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(overall_tokens)
    trigram_finder.ngram_fd.tabulate(10)

    quadgram_finder = nltk.collocations.QuadgramCollocationFinder.from_words(overall_tokens)
    quadgram_finder.ngram_fd.tabulate(10)

    sia = SentimentIntensityAnalyzer()

    def tweet_find_nltk_polarity(x: pd.Series):
        senti = sia.polarity_scores(x[text_col_name])
        return senti['compound']

    tweets_csv[ref_sentiment_name] = tweets_csv.apply(tweet_find_nltk_polarity, axis=1)

    # pad features
    def pad_tokens(x: pd.Series):
        tokens = x[tokenized_col_name]
        padding = [0] * (50 - len(tokens))
        return padding + tokens

    tweets_csv.loc[:, tokenized_col_name] = tweets_csv.apply(pad_tokens, axis=1)



    tokens_full_series = tweets_csv[tokenized_col_name]
    polarity_full_series = tweets_csv[polarity_label_name]
    # tokens_full_series.to_list()

    tokens_full_nparr = np.asarray(tokens_full_series.to_list(), dtype=int)
    np.random.shuffle(tokens_full_nparr)

    train_valid_split_point = int(len(tokens_full_nparr) * 0.8)
    valid_test_split_point = int(len(tokens_full_nparr) * 0.9)

    train_tokens = tokens_full_nparr[: train_valid_split_point]
    valid_tokens = tokens_full_nparr[train_valid_split_point: valid_test_split_point]
    test_tokens = tokens_full_nparr[valid_test_split_point:]

    polarity_full_nparr = np.asarray(polarity_full_series.to_list(), dtype=int)
    np.random.shuffle(polarity_full_nparr)

    train_polarity = polarity_full_nparr[: train_valid_split_point]
    valid_polarity = polarity_full_nparr[train_valid_split_point: valid_test_split_point]
    test_polarity = polarity_full_nparr[valid_test_split_point:]

    full_data = TensorDataset(torch.from_numpy(tokens_full_nparr), torch.from_numpy(polarity_full_nparr))
    train_data = TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_polarity))
    valid_data = TensorDataset(torch.from_numpy(valid_tokens), torch.from_numpy(valid_polarity))
    test_data = TensorDataset(torch.from_numpy(test_tokens), torch.from_numpy(test_polarity))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train, val, test = get_dataloaders(16, "../dataset/biden_tweets_labeled.csv")
    samp  = []
    cnt = 0
    for data in train:
        samp.extend(data[1])
