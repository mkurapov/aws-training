import pandas as pd
import sys
import numpy as np
import nltk
import string
import os
import time

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/cleaned.csv', nrows=4)
df2 = pd.read_csv('data/cleaned.csv', nrows=1, header=None)

# For example an ngram_range of (1, 1) means only unigrams,
# (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
vectorizer = TfidfVectorizer(ngram_range=(1, 1))

X = vectorizer.fit_transform(df['text'])
print(X.toarray())
y = df['tag'].map({'-': 1, '+': 2, '0' : 0})

# print(vectorizer.get_feature_names())
# print(X.shape)

lin_clf = svm.LinearSVC(multi_class='ovr')
lin_clf.fit(X, y)

x_test = vectorizer.fit_transform(df2[1])
print(x_test)
print(lin_clf.predict(x_test))
# vectorised_training = vectorise.fit_transform(training_set)
# vectorised_testing = vectorise.fit_transform(testing_set)
