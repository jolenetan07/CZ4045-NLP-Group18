# CZ4045 NLP Group18 Sentiment Analysis Project

## File Structure
| datasets
|||| preprocess_data.ipynb
|||| biden_tweets_raw.csv
|||| biden_tweets_processed.csv
|||| biden_tweets_labeled.csv
|||| biden_tweets_labeled_processed.csv
|||| biden_tweets_labeled_train.csv
|||| biden_tweets_labeled_test.csv

| classifiers
|||| vader_classifier.ipynb
|||| textblob_classifier.ipynb
|||| lsvc_classifier.ipynb
|||| sgdc_classifier.ipynb
|||| xgb_classifier.ipynb
|||| cnb_classifier.ipynb

| results
|||| vader_results_full.csv
|||| vader_results_test_labeled.csv
|||| textblob_results_full.csv
|||| textblob_results_test_labeled.csv
|||| lsvc_results_full.csv
|||| lsvc_results_test_labeled.csv
|||| sgdc_results_full.csv
|||| sgdc_results_test_labeled.csv
|||| xgb_results_full.csv
|||| xgb_results_test_labeled.csv
|||| cnb_results_full.csv
|||| cnb_results_test_labeled.csv

| innovations
|||| tune-xgboost
|||||||| tune_xgb_classifier.ipynb

|||| choose-bayes
|||||||| choose_naivebayes_classifier.ipynb

|||| bigrams
|||||||| classifiers
|||||||||||||||| bigram_lsvc_classifier.ipynb
|||||||||||||||| bigram_sgdc_classifier.ipynb
|||||||||||||||| bigram_xgb_classifier.ipynb
|||||||||||||||| bigram_cnb_classifier.ipynb
|||||||| results
||||||||||||||| bigram_lsvc_results_full.csv
||||||||||||||| bigram_lsvc_results_test_labeled.csv
||||||||||||||| bigram_sgdc_results_full.csv
||||||||||||||| bigram_sgdc_results_test_labeled.csv
||||||||||||||| bigram_xgb_results_full.csv
||||||||||||||| bigram_xgb_results_test_labeled.csv
||||||||||||||| bigram_cnb_results_full.csv
||||||||||||||| bigram_cnb_results_test_labeled.csv

|||| trigrams
|||||||| classifiers
|||||||||||||||| trigram_lsvc_classifier.ipynb
|||||||||||||||| trigram_sgdc_classifier.ipynb
|||||||||||||||| trigram_xgb_classifier.ipynb
|||||||||||||||| trigram_cnb_classifier.ipynb
|||||||| results
||||||||||||||| trigram_lsvc_results_full.csv
||||||||||||||| trigram_lsvc_results_test_labeled.csv
||||||||||||||| trigram_sgdc_results_full.csv
||||||||||||||| trigram_sgdc_results_test_labeled.csv
||||||||||||||| trigram_xgb_results_full.csv
||||||||||||||| trigram_xgb_results_test_labeled.csv
||||||||||||||| trigram_cnb_results_full.csv
||||||||||||||| trigram_cnb_results_test_labeled.csv


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
    - classification report (precision, recall, f1-score)
    - sentiment counts
    - sentiment counts pie chart
    - sentiment counts bar plot


## Innovations
* xgboost hyperparameters tuning
* choosing best variation of naive bayes
* bigram and trigram on supervised models 

