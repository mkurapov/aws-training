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

sentimentDictionary = {
    "Neutral": "0",
    "Positive": "+",
    "Negative": "-",
    "tag": "0"
}


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def cleanup1(row):
    if (pd.isna(row['text'])):
        return row

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

    row['text'] = tweet_text
    return row


def filter_word(word):
    if len(word) == 1 or word == 'https':
        return False

    return True

# minimize tags, remove single char words, https, duplicate words
def cleanup2(row):
    if (pd.isna(row['text'])):
        return row

    tweet_text = row['text']
    if (tweet_text == '' or tweet_text is None):
        return row

    stripped = tweet_text.strip()
    clean_string = ' '.join(unique_list(filter(filter_word, stripped.split())))
    row['text'] = clean_string

    return row

def cleanup(row):
    print(row.name)
    cleanup1(row)
    cleanup2(row)
    return row

def process_df(df):
    try:
        df = df.apply(cleanup, axis=1)
        df.to_csv(f"output/test_data.csv",
                  sep=",", header=True, index=False)
    except Exception as e:
        print(e)

CHUNK_SIZE = 100000
count = CHUNK_SIZE

df = df[['text']]
process_df(df)

for df in df_chunks:
    tic = time.perf_counter()
    process_df(df, count)
    toc = time.perf_counter()

    print(f"Processed {count} rows in {toc - tic:0.4f} sec")
    count = count + CHUNK_SIZE


### Copy column over