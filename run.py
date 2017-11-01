from __future__ import print_function
import os
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

path = 'data/sms.tsv'
sms = pd.read_table(path, header=None, names=['label', 'message'])

sms.shape

# examine the first 10 rows
sms.head(10)

# examine the class distribution
sms.label.value_counts()

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

# check that the conversion worked
sms.head(10)

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)
print('\n')

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print('\n')


# Vectorizing our dataset
# instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)

# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix
X_train_dtm

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# Building and evaluating a model
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# print the confusion matrix
accuracy_score = metrics.accuracy_score(y_test, y_pred_class)
print ('Accuracy for Multinomial Naive Bayes is {0}'.format(accuracy_score))

# calculate predicted probabilities for X_test_dtm
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob

# calculate AUC
auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
print ('Area Under ROC curve for Multinomial Naive Bayes is {0}'.format(auc_score))
print('\n')

# Comparing models

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# train the model using X_train_dtm
logreg.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)

# calculate predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob

# calculate accuracy
accuracy_score = metrics.accuracy_score(y_test, y_pred_class)
print ('Accuracy for Logistic Regression is {0}'.format(accuracy_score))

# calculate AUC

auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
print ('Area Under ROC curve for Logistic Regression is {0}'.format(auc_score))
print('\n')
# Examining a model for further insight
# calculate the approximate "spamminess" of each token

# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


#
tokens = X_train_tokens[0:50]
print('examine the first 50 tokens:\n {0}'.format(tokens))
print('\n')

# examine the last 50 tokens
tokens = X_train_tokens[-50:]
print('examine the last 50 tokens:\n {0}'.format(tokens))
print('\n')

# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_

# rows represent classes, columns represent tokens
nb.feature_count_.shape

# number of times each token appears across all HAM messages
ham_token_count = nb.feature_count_[0, :]
ham_token_count

# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1, :]
spam_token_count

# create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count, 'spam':spam_token_count}).set_index('token')
tokens.head()

# examine 5 random DataFrame rows
tokens.sample(5, random_state=6)

# Naive Bayes counts the number of observations in each class
nb.class_count_

# Before we can calculate the "spamminess" of each token, we need to avoid dividing by zero and account for the class imbalance.
# add 1 to ham and spam counts to avoid dividing by 0
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1
tokens.sample(5, random_state=6)


# convert the ham and spam counts into frequencies
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]
tokens.sample(5, random_state=6)


# calculate the ratio of spam-to-ham for each token
tokens['spam_ratio'] = tokens.spam / tokens.ham
tokens.sample(5, random_state=6)


# examine the DataFrame sorted by spam_ratio
tokens.sort_values('spam_ratio', ascending=False)

# look up the spam_ratio for a given token
spam_ratio = tokens.loc['dating', 'spam_ratio']
print ('spam_ratio for word "dating" is {0}'.format(spam_ratio))
print('\n')
