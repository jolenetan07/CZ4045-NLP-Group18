from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import numpy as np
from torch import nn
import torch

# %%
df = pd.read_csv(('dataset/biden_tweets_labeled.csv'))
df.drop(columns=['Unnamed: 0'], inplace=True)
subjectivity = df['subjectivity'].to_numpy()
# 0:neg, 1:pos, 2:neutral
polarity = df['polarity'].to_numpy()



def remap_label(polarity):
    polarity_onehot = np.zeros((len(polarity), 3))
    for (i, p) in enumerate(polarity):
        if p == 0:
            polarity_onehot[i][0] = 1
        elif p == 1:
            polarity_onehot[i][2] = 1
        elif p == 2:  # put neutral in the middle
            polarity_onehot[i][1] = 1
        else:
            raise ValueError('polarity value not in range')
    return polarity_onehot


#  roBERTa is  0 neg, 1 neutral, 2 pos, remap the labels
polarity_onehot = remap_label(polarity)

# %%
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

# %%
roberta_loss = []
correct = 0
for (i, tweet) in enumerate(df['Text'].to_numpy()):
    tweet_words = []
    for word in tweet.split():
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = ' '.join(tweet_words)
    # print(tweet_proc)
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    scores = softmax(output[0][0].detach().numpy())
    crit = nn.CrossEntropyLoss()
    roberta_loss.append(crit(torch.tensor(scores), torch.tensor(polarity_onehot[i])).item())
    # print("roberta:",labels[scores.argmax()], "label:",labels[polarity_onehot[i].argmax()])
    if scores.argmax() == polarity_onehot[i].argmax():
        correct += 1
print("roberta accuracy:", correct / len(roberta_loss))











