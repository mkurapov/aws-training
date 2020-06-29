import pandas as pd
import sys
import numpy as np
import nltk
import string
import os
import time

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/cleaned.csv', nrows=10)

# For example an ngram_range of (1, 1) means only unigrams,
# (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
vectorizer = TfidfVectorizer(ngram_range=(1, 1))

X = vectorizer.fit_transform(df['text'])
y = ['Positive', 'Neutral', 'Negative']

print(vectorizer.get_feature_names())
print(X.shape)


lin_clf = svm.LinearSVC(multi_class='ovr')
lin_clf.fit(X, y)

print(lin_clf.predict([[-0.8, -1]]))
# vectorise = TfidVectorizer(ngram_range = (0,1,2))
# vectorised_training = vectorise.fit_transform(training_set)
# vectorised_testing = vectorise.fit_transform(testing_set)
