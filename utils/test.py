import pandas as pd
import numpy as np


#%%
df = pd.read_csv('dataset/biden_tweets_clean.csv', sep=',', header=None)

#%%
df2 = df.drop(columns=[0], inplace=False)
print(df2.count())
df2.drop_duplicates(inplace=True)
tweets = df2.values.tolist()
print(len(tweets))


