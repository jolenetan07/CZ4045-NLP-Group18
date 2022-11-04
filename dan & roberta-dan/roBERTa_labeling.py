from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import numpy as np
from dataset.data_processing_pipeline import *
from dataset.data_rough_processing import get_processed_df

from torch import nn
import torch

# Here, we use a pretrained model roBERTa to to auto label out dataset


def remap_label(predict):
    """
    the output of roberta is  0 for neg, 1 for neutral and 2 for pos
    our manually labelled data is 0 for neg, 1 for pos and 2 for neutral
    remap the labels
    """
    remapped = np.zeros(len(predict))
    for (i, p) in enumerate(predict):
        if p == 0:
            remapped[i] = 0
        elif p == 1:
            remapped[i] = 2
        elif p == 2:  # put neutral in the middle
            remapped[i] = 1
        else:
            raise ValueError('polarity value not in range')
    return remapped


def roBERTa():
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    tokenizer.model_max_length = 512
    return model, tokenizer


def get_roBERTa_label(df):
    roberta, tokenizer = roBERTa()
    pipeline = DataProcessingPipeline([
        ReplaceWords(),
        # replace_emoji_with_text,
    ])
    text_data = df['Text'].apply(pipeline).to_numpy()
    predicted_labels = []

    for (i, tweet) in enumerate(text_data):
        encoded_tweet = tokenizer(tweet, return_tensors='pt')
        if encoded_tweet['input_ids'].shape[1] > 512:
            print(tweet)

        output = roberta(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
        scores = softmax(output[0][0].detach().numpy())
        predicted_labels.append(scores.argmax())

    return remap_label(predicted_labels)


def roberta_labeled_df():
    df = get_processed_df('dataset/biden_tweets_labeled.csv',include_polarity=True)
    df['roberta_labeled'] = get_roBERTa_label(df)
    df.to_csv('dataset/biden_tweets_manually_labeled_roberta_labeled.csv')
    return df


df = roberta_labeled_df()

