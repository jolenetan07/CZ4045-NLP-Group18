# CZ4045 NLP Group18 Sentiment Analysis Project

## File Structure
| datasets\
|||| crawl_data.ipynb\
|||| preprocess_data.ipynb\
|||| biden_tweets_raw.csv\
|||| biden_tweets_processed.csv\
|||| biden_tweets_labeled.csv\
|||| biden_tweets_labeled_processed.csv\
|||| biden_tweets_labeled_train.csv\
|||| biden_tweets_labeled_test.csv\

| classifiers\
|||| vader_classifier.ipynb\
|||| textblob_classifier.ipynb\
|||| lsvc_classifier.ipynb\
|||| xgb_classifier.ipynb\
|||| cnb_classifier.ipynb\
|||| results
|||||||| vader_results_full.csv\
|||||||| vader_results_test_labeled.csv\
|||||||| textblob_results_full.csv\
|||||||| textblob_results_test_labeled.csv\
|||||||| lsvc_results_full.csv\
|||||||| lsvc_results_test_labeled.csv\
|||||||| xgb_results_full.csv\
|||||||| xgb_results_test_labeled.csv\
|||||||| cnb_results_full.csv\
|||||||| cnb_results_test_labeled.csv\

| deep learning models\
/* TODO :: insert file structures here */

| ui\
|||| index.html\
|||| styles.css\
|||| scripts.js\

| innovations\
|||| tune-xgboost\
|||||||| tune_xgb_classifier.ipynb\

|||| choose-bayes\
|||||||| choose_naivebayes_classifier.ipynb\

|||| bigrams\
|||||||| classifiers\
|||||||||||||||| bigram_lsvc_classifier.ipynb\
|||||||||||||||| bigram_xgb_classifier.ipynb\
|||||||||||||||| bigram_cnb_classifier.ipynb\
|||||||||||||||| results\
|||||||||||||||||||| bigram_lsvc_results_full.csv\
|||||||||||||||||||| bigram_lsvc_results_test_labeled.csv\
|||||||||||||||||||| bigram_xgb_results_full.csv\
|||||||||||||||||||| bigram_xgb_results_test_labeled.csv\
|||||||||||||||||||| bigram_cnb_results_full.csv\
|||||||||||||||||||| bigram_cnb_results_test_labeled.csv\

|||| trigrams\
|||||||| classifiers\
|||||||||||||||| trigram_lsvc_classifier.ipynb\
|||||||||||||||| trigram_xgb_classifier.ipynb\
|||||||||||||||| trigram_cnb_classifier.ipynb\
|||||||||||||||| results\
|||||||||||||||||||| trigram_lsvc_results_full.csv\
|||||||||||||||||||| trigram_lsvc_results_test_labeled.csv\
|||||||||||||||||||| trigram_xgb_results_full.csv\
|||||||||||||||||||| trigram_xgb_results_test_labeled.csv\
|||||||||||||||||||| trigram_cnb_results_full.csv\
|||||||||||||||||||| trigram_cnb_results_test_labeled.csv\


## Data Collection 
*** datasets folder ***
* 'crawl_data.ipynb' : python script to crawl tweet text data using snscrape


## Data Preprocessing
*** datasets folder ***
* 'preprocess_data.ipynb' : python script to clean and preprocess data using various techniques & perform exploratory data analysis on both full and labeled datasets 


## Dataset
*** datasets folder ***
* 'biden_tweets_raw.csv' : tweet data crawled using snscrape 

* 'biden_tweets_processed.csv' : clean tweet data and append preprocessed tweet text data 

* 'biden_tweets_labeled.csv' : extracted 10% of full dataset to append manual labels on subjectivity and polarity

* 'biden_tweets_labeled_processed.csv' : append preprocessed labeled tweet text data  

* 'biden_tweets_labeled_train.csv' : extract 80% of preprocessed labeled tweet text data  to use as train dataset for models

* 'biden_tweets_labeled_test.csv' : extract 20% preprocessed labeled tweet text data  to use as test dataset for models


## Sentiment Analysis (Classification)
*** classifiers folder ***
### vadersentiments
* 'vader_classifier.ipynb' : python script for sentiment analysis with vader classifier

### textblob
* 'textblob_classifier.ipynb' : python script for sentiment analysis with textblob classifier

### linear support vector 
* 'lsvc_classifier.ipynb' : python script for sentiment analysis with linearSVC classifier

### xgboost
* 'xgb_classifier.ipynb' : python script for sentiment analysis with XGBoost classifier

### complement naive bayes
* 'cnb_classifier.ipynb' : python script for sentiment analysis with ComplementNB classifier

*** results folder ***
* '<model>_results_full.csv : append predicted polarity by <model> classifier to full dataset
* '<model>_results_test_labeled.csv : append predicted polarity by <model> classifier to labeled test dataset

*** deep learning folder ***
/* TODO :: insert file descriptions here */

## Classification Dashboard (UI)
*** ui folder ***
* 'index.html'
* 'styles.css'
* 'script.js'


## Innovations
*** innovations folder ***
### xgboost hyperparameters tuning
*** tune-xgboost folder ***
* 'tune_xgb_classifier.ipynb' : python script to find best hyperparameters for sentiment analysis with XGBoost classifier

### choosing best variation of naive bayes
*** choose-naivebayes folder ***
* 'choose_naivebayes_classifier.ipynb' : python script to find best variation of naive bayes classifier for sentiment analysis 

### bigram on supervised models 
*** bigrams -> classifier folder ***
#### linear support vector 
* 'bigram_lsvc_classifier.ipynb' : python script for sentiment analysis with linearSVC classifier with bi-gram

#### xgboost
* 'bigram_xgb_classifier.ipynb' : python script for sentiment analysis with XGBoost classifier with bi-gram

#### complement naive bayes
* 'bigram_cnb_classifier.ipynb' : python script for sentiment analysis with ComplementNB classifier with bi-gram


*** results folder ***
* 'bigram_<model>_results_full.csv : append predicted polarity by <model> classifier with bi-gram to full dataset
* 'bigram_<model>_results_test_labeled.csv : append predicted polarity by <model> classifier wwith bi-gram to labeled test dataset

### trigram on supervised models
*** trigrams -> classifier folder ***
#### linear support vector 
* 'trigram_lsvc_classifier.ipynb' : python script for sentiment analysis with linearSVC classifier with tri-gram

#### xgboost
* 'trigram_xgb_classifier.ipynb' : python script for sentiment analysis with XGBoost classifier with tri-gram

#### complement naive bayes
* 'trigram_cnb_classifier.ipynb' : python script for sentiment analysis with ComplementNB classifier with tri-gram

*** results folder ***
* 'trigram_<model>_results_full.csv : append predicted polarity by <model> classifier with tri-gram to full dataset
* 'trigram_<model>_results_test_labeled.csv : append predicted polarity by <model> classifier wwith tri-gram to labeled test dataset



