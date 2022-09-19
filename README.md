# Predicticting-Twitter-Sentiment

## Project Overview

South by Southwest wants a predictive model that can predict the sentiment of a tweet as either positive, negative, or neutral. They want to use this model on tweets about their event so they can have a better understanding of what people are looking forward to and what they are upset about. It is important for SXSW that this model does not falsely identify tweets as either positive or negative so I will use F1 Score and Accuracy Score as the main metrics for evaluating this model. In this project, I created a Logistic Regression model using a TFIDF Vectorizer on the tweets and One Hot Encoding the column that displays what company the tweet is talking about whether it be 'Apple', 'Google', or 'Unknown'. This model has an accuracy and F1 score of 90% and 88% on test data. In this repository you will find 2 jupyter notebooks. One for the data cleaning and exploration/analysis and the other for modeling. An important note about the modeling notebook is it contains two grid searchs which can take a long time to run depending on the computer you have. So keep that in mind when running that notebook. This repository also contains a data folder containing two csv files I used for this project.

## Business and Data Understanding

### Business Problem

South by Southwest wants a predictive model that can predict the sentiment of a tweet as either positive, negative, or neutral. They want to use this model on tweets about their event so they can have a better understanding of what people are looking forward to and what they are upset about. Once they identified new tweets as either positive, negative, or neutral they can run sentiment analysis on these tweets to find out the words and phrases associated with those emotions. This can help with planning events and advertising for their conference because they will have a better idea of audience reception and attendance towards certain events and products. I wanted to focus on models that have both high accuracy and recall scores as our stakeholder does not want tweets to be mislabeled.

### Data Understanding

The dataset used is from data.world and contains over 9,000 tweets containing the hashtag #sxsw. The dataset contains 3 columns: context of the tweet, what product or company it is directed at, and the emotion of the tweet. As part of the cleaning process, I created a new column, 'company', which contains the company that the tweet is referencing. In the case of this dataset there companies mentioned were either 'Apple', 'Google' and 'Unknown'. 'Unknown' being if the tweet did not have a clear company in its context. From there I cleaned the tweet text by removing any mentions and hashtags in the tweets and making the whole string lower case. After cleaning the tweet, a new problem arised. There were duplicate tweets that each had a different sentiment associated with them. For example, the same tweet could be labeled as both 'positive' and 'negative'. This logically did not make sense because a tweet can only have one sentiment to it. For this reason, I removed all duplicate tweets that had multiple sentiments to them.

## Modeling and Evaluation

My goal was to create a classification model to predict tweet sentiment and I focused on having both high accuracy and F1 scores because I wanted to minimize the amount of False Negatives and False Positives. I set a baseline model using a Dummy Classification model that would predict the most frequent value. The baseline model had an accuracy score of 59.8% and F1 score of 44.8% on the train data. 

After the baseline model, I decided to create more classification models while testing to see if Count Vectorizer or TFIDF Vectorizer worked better. The performance of the models did not vary much between the two vectorizers. The models I created after the baseline model were logistic regression, decision tree, random forest, gradient boost, adaboost, and xgboost. The models had accuracy scores up to 70%. After testing each vectorizer, I decided to One Hot Encode the company column and train a model using just the One Hot Encoder. The models using the One Hot Encoder performed far better than the vectorizers. After I used a combination of One Hot Encoder and TFIDF Vectorizer to train the models. These models performed the best with accuracy scores up to 90%. 

I used GridSearchCV to test different parameters for the TFIDF Vectorizer on the Logistic Regression model. The parameters I searched were max_features, n_gram_range, and min_df. The grid searches proved effective as I ended up with a well trained Logistic Regression model.

### Final Model - Logistic Regression

My final model was the best estimator Logistic Regression model from the grid search. On the test set, the model had the following performance metrics:

Accuracy score: 0.902

F1 score: 0.884

Precision score: 0.896

Recall score: 0.902

![image](https://user-images.githubusercontent.com/108245743/190938746-6913713a-b338-4b5e-8c53-e2fabc8d195d.png)

## Conclusion

My final classification model for South by Southwest produced an accuracy score and F1 score of 90.2% and 88.4% on the test data. This model will work well with what South by Southwest is trying to accomplish. South by Southwest will be able to use this predictive model to identify the emotion of tweets about their conference. Using the predicted emotions, South by Southwest can analyze the words and phrases asscociated with the emotions to better understand audience reception and anticipation of the event.

## Repository Structure

```
├── Data
│   ├── tweets_clean.csv
│   ├── tweet_product_company.csv
├── PDFs
│   ├── Cleaning_and_eda.pdf
│   ├── Modeling.pdf
│   ├── Presentation.pdf
├── .gitignore
├── Cleaning_and_eda.ipynb
├── Modeling
└── README.md
```
