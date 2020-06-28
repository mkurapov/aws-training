import pandas as pd
import sys
import numpy as np
import nltk
import spacy
import string
import os
from nltk.corpus import stopwords
import preprocessor as p
import time

nlp = spacy.load('en_core_web_sm')
en_stopwords = stopwords.words("English")
extension = ["a", "I", "&amp;", "https", "http"]
strip_punctation_table = str.maketrans('', '', string.punctuation)


def cleanup(row):
    tweet_text = row['text']
    tweet_text = p.clean(tweet_text)
    tweet_text = tweet_text.lower()

    clean_words = []
    for word in tweet_text.split():
        if ((word not in en_stopwords) and (word not in extension)):
            clean_word = word.translate(strip_punctation_table)
            if (clean_word != ''):
                clean_words.append(clean_word)

    clean_string = ' '.join(clean_words)
    tweet_text = ' '.join(word.lemma_ for word in nlp(clean_string) if word.lemma_ !=
                          '-PRON-')

    return tweet_text


CHUNK_SIZE = 10000
count = CHUNK_SIZE
df_chunks = pd.read_csv('data/Training_Data_TAGGED.csv',
                        skiprows=range(1, 1500000),
                        chunksize=CHUNK_SIZE)

for df in df_chunks:
    df.rename(columns={df.columns[0]: "tweet_id"}, inplace=True)
    try:
        df['text'] = df.apply(cleanup, axis=1)
        df.to_csv(f"output/chunk{count}.csv",
                  sep=",", header=True, index=False)
    except Exception as e:
        print(e)

    print(f"Processed {count} rows")
    count = count + CHUNK_SIZE
