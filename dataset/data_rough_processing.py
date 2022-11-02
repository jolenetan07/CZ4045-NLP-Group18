# -*- coding: utf-8 -*-

# -- Sheet --


from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import pandas as pd
import re

import nltk

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

sentiment_name = "Sentiment"
text_col_name = "Text"
subjectivity_label_name = "subjectivity"
polarity_label_name = "polarity"
token_col_name = "Raw tokens"
tokenized_col_name = "Tokenized"
length_col_name = "Token length"
ref_sentiment_name = "NLTK ref sentiment"

use_csv_col_as_idx = False
# data_path = "biden_tweets_labeled.csv"
data_path = "dataset/biden_tweets_clean.csv"
# columns_to_read = [text_col_name, subjectivity_label_name, polarity_label_name]
columns_to_read = [text_col_name]

unk_word_name = "unknown word"
unknown_word_id = -1

remove_at_tags_in_tweets = True
label_map_dict = {2: 2, 1: 1, 0: 0}

truncate_length = 50
truncate_to_unknown_corpus_length_limit = 5000

append_nltk_reference = True
pad_features = True

# tweets_csv = pd.read_csv(data_path)
# if tweets_csv.columns[0] == "Unnamed: 0":
print(f"reading data from path {data_path}")
if use_csv_col_as_idx:
    print(f"first column as index, reading csv")
    tweets_csv = pd.read_csv(data_path, index_col=[0])
else:
    print(f"first column is named, fall back to specify used_cols")
    tweets_csv = pd.read_csv(data_path, usecols=columns_to_read)

stopwords = nltk.corpus.stopwords.words("english")

overall_tokens = []

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


def apply_self_mapping_of_label(x: pd.Series):
    return label_map_dict[x[polarity_label_name]]


if remove_at_tags_in_tweets:
    tweets_csv[text_col_name] = tweets_csv.apply(remove_at_tags, axis=1)

tweets_csv[token_col_name] = tweets_csv.apply(tweet_en_tokenize, axis=1)

# tweets_csv[polarity_label_name] = tweets_csv.apply(apply_self_mapping_of_label, axis=1)

tweets_csv[length_col_name] = tweets_csv.apply(lambda x: len(x[token_col_name]), axis=1)

tweets_csv = tweets_csv[tweets_csv[length_col_name] <= truncate_length]

tweet_freq_dict = nltk.FreqDist(overall_tokens)
print(type(tweet_freq_dict))
tweet_freq_dict.tabulate(25)

vocab_to_int_encoding = {pair[1]: pair[0] + 1 for pair in enumerate(tweet_freq_dict)}
# print(len(vocab_to_int_encoding))
# print(type(vocab_to_int_encoding))


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

if append_nltk_reference:
    sia = SentimentIntensityAnalyzer()


    def tweet_find_nltk_polarity(x: pd.Series):
        senti = sia.polarity_scores(x[text_col_name])
        return senti['compound']


    tweets_csv[ref_sentiment_name] = tweets_csv.apply(tweet_find_nltk_polarity, axis=1)

if pad_features:
    # pad features
    def pad_tokens(x: pd.Series):
        tokens = x[tokenized_col_name]
        padding = [0] * (truncate_length - len(tokens))
        return padding + tokens


    tweets_csv.loc[:, tokenized_col_name] = tweets_csv.apply(pad_tokens, axis=1)




def get_processed_df():
    return tweets_csv

