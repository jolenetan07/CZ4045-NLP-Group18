# CZ4045-NLP-Group18

## Dataset Info
* biden_tweets_raw.csv
    - raw data obtained using snscrape to crawl tweets

* biden_tweets_clean.csv (run clean_data.ipynb)
    - initial cleaning of data
    - remove non-English tweets
    - remove duplicate tweets
    - drop all columns except tweet text

* biden_tweets_labeled.csv
    - manually labeled 10% of tweets from biden_tweets_clean.csv (5% from top, 5% from bottom)
    - to be used as training and/or evaluation set 
    - subjectivity - 0: neutral, 1: opinionated
    - polarity - 0: negative, 1: positive, 2: negative

* biden_tweets_processed.csv (run preprocess_data.ipynb)
    - final cleaning of data
    - preprocess text data using various techniques