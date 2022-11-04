from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

import pandas as pd


tweet = "Well, I didn't vote for Biden nor Trump, 1 of them was going to win. 🤷🏻‍♀️If it bothers you that I like a less toxic tone... sorry... not sorry. 😎Have a good night. 👌"


tweet_words = []

for word in tweet.split(' '):
     if word.startswith('@') and len(word) > 1:
          word = '@user'

     elif word.startswith('http'):
          word = "http"
     tweet_words.append(word)

tweet_proc = ' '.join(tweet_words)

# download the model and tokenizer

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)

tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

scores = softmax(output[0][0].detach().numpy())
print(output)

# print(encoded_tweet)



def load_unlabeled_data(path):
     df = pd.read_csv(path, sep=',', header=None)


