# CZ4045 NLP Group18 Sentiment Analysis Project

## Data Collection
*** TODO :: upload snscrape python script ***


## Data Preprocessing
* 'preprocess_data.ipynb' : python script to clean and preprocess data using various techniques & perform exploratory data analysis on dataset (both full dataset and labeled dataset included)


## Dataset
* 'biden_tweets_raw.csv' : tweet data crawled using snscrape 

* 'biden_tweets_processed.csv' : clean tweet data and append preprocessed tweet text data 

* 'biden_tweets_labeled.csv' : extracted 10% of full dataset to append manual labels on subjectivity and polarity

* 'biden_tweets_labeled_processed.csv' : append preprocessed labeled tweet text data  

* 'biden_tweets_labeled_train.csv' : extract 80% of preprocessed labeled tweet text data  to use as train dataset for models

* 'biden_tweets_labeled_test.csv' : extract 20% preprocessed labeled tweet text data  to use as test dataset for models


## Sentiment Analysis (Classification)
* unsupervised models
    - vadersentiments
    - textblob

* supervised models
    - linear support vector 
    - stochastic gradient descent 
    - xgboost
    - complement naive bayes


## Classification UI
* overview 
    - accuracy
    - sentiment counts 

* model specific
    - classification report
    - sentiment counts
    - sentiment counts pie chart
    - sentiment counts bar plot


## Innovations
* xgboost hyperparameters tuning
* choosing best variation of naive bayes
* bigram and trigram on supervised models 

