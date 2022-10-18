import pandas as pd
import torch

tweets_csv = pd.read_csv("./dataset/biden_tweet_text.csv")
if tweets_csv.columns[0] == "Unnamed: 0":
    print(f"first column as index, reading csv")
    tweets_csv = pd.read_csv("./dataset/biden_tweet_text.csv", index_col=[0])
else:
    print(f"first column is named, fall back to specify used_cols")
    tweets_csv = pd.read_csv("./dataset/biden_tweet_text.csv", usecols=["Tweet Text", "Sentiment"])



print(tweets_csv.head())



