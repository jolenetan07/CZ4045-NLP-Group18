import string

import pandas as pd
import torch
import re
import matplotlib.pyplot as plt

sentiment_name = "Sentiment"
text_col_name = "Tweet Text"

use_csv_col_as_idx = False

tweets_csv = pd.read_csv("./dataset/biden_tweet_text.csv")
# if tweets_csv.columns[0] == "Unnamed: 0":
if use_csv_col_as_idx:
    print(f"first column as index, reading csv")
    tweets_csv = pd.read_csv("./dataset/biden_tweet_text.csv", index_col=[0])
else:
    print(f"first column is named, fall back to specify used_cols")
    tweets_csv = pd.read_csv("./dataset/biden_tweet_text.csv", usecols=["Tweet Text", "Sentiment"])

# print(tweets_csv.columns)
# for i in range(30):
#     print(tweets_csv["Tweet Text"][i])

tweets_csv["Sentiment"] = tweets_csv["Sentiment"].apply(lambda x: 0 if re.match(".*negative.*", x) else 1)

# print(tweets_csv.head())

# reviews_len = [len(x) for x in tweets_csv["Tweet Text"]]
# pd.Series(reviews_len).hist(bins=400)
# # plt.show()
# print(pd.Series(reviews_len).describe())


tweets_csv["length"] = tweets_csv["Tweet Text"].apply(lambda x: len(x))

print(tweets_csv["length"].describe())
tweets_csv = tweets_csv[300 > tweets_csv["length"]]
tweets_csv = tweets_csv[25 < tweets_csv["length"]]

print(tweets_csv["length"].describe())


def construct_corpus(text_container) -> dict[str, int]:
    corpus = {}

    def add_corpus(elem: str):
        if elem in corpus:
            corpus[elem] += 1
        else:
            corpus[elem] = 1
    considered_punc = ('?', '!', '.')
    for text in text_container:
        text: str
        tokens = text.split(sep=' ')
        for token in tokens:
            if token in corpus:
                corpus[token] += 1
            else:
                if re.match("[0-9]+", token):
                    add_corpus("number")
                elif re.match(".*\s+.*", token):
                    pass
                elif token in string.punctuation:
                    if token not in considered_punc:
                        pass
                    else:
                        add_corpus(token)
                # elif re.match(".*[!?.]", token):
                #     punc = token[-1]
                #     add_corpus(punc)
                #     add_corpus(token[:-1])
                else:
                    corpus[token] = 1
    return corpus


corpus = construct_corpus(tweets_csv[text_col_name])
corpus_df = pd.DataFrame.from_dict(corpus, orient='index')
print(corpus_df.describe(percentiles=[x / 10 for x in range(1, 10)]))

corpus_df.insert(0, 'id', range(len(corpus_df)))
# corpus_df.reset_index(inplace=True)
# corpus_df.rename(columns={"index":"word", 0:"count"}, inplace=True)
print(corpus_df.head())


def _text_to_embeddings(x: pd.Series):
    # print(f"in apply, x={x}")
    cur_row = x.copy()
    cur_text = x[text_col_name]
    tokens = cur_text.split(sep=' ')
    result = []
    for token in tokens:
        if token not in corpus:
            continue
        id = corpus_df.loc[token]["id"]
        result.append(id)
    cur_row[text_col_name] = tuple(result)
    return cur_row


tweets_csv_tokenized = tweets_csv.apply(_text_to_embeddings, axis=1)

print(tweets_csv_tokenized.head())

print(corpus_df.head(30))
