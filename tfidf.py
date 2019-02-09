import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import operator
import collections
import nltk
import re
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.probability import LaplaceProbDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sys import argv
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    train = pd.read_csv('./Headline_Trainingdata.csv')
    test = pd.read_csv('./Headline_Testingdata.csv')
    # test = pd.read_csv('./fake.csv')

    return train, test

# extraction features from corpus


def split_label_feats(lfeats, split=0.75):
    train_feats = []
    # test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        # test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats

# split corpus into training set & dev-test set
# very_negative:0, negative:1, neutral:2, positive:3, very positive:4.

def addRows(df, needSize, size):
    quotient = needSize // size
    remainder = needSize % size

    whole = df.copy()
    if quotient != 0:
        whole = pd.concat([df.copy()] * (quotient+1), ignore_index=True)

    X = df.sample(n=remainder)

    result = pd.concat([whole, X.copy()], ignore_index=True)

    return result

def rebuild_corpus(corpus):
    veryNeg = shuffle(corpus.loc[corpus['sentiment'] == 0])
    negative = shuffle(corpus.loc[corpus['sentiment'] == 1])
    neutral = shuffle(corpus.loc[corpus['sentiment'] == 2])
    positive = shuffle(corpus.loc[corpus['sentiment'] == 3])
    veryPos = shuffle(corpus.loc[corpus['sentiment'] == 4])

    veryNegSize = veryNeg.shape[0]
    negativeSize = negative.shape[0]
    neutralSize = neutral.shape[0]
    positiveSize = positive.shape[0]
    veryPosSize = veryPos.shape[0]

    maxSize = max(veryNegSize, negativeSize, neutralSize, positiveSize, veryPosSize)

    veryNegAdd = maxSize - veryNegSize
    negativeAdd = maxSize - negativeSize
    neutralAdd = maxSize - neutralSize
    veryPosAdd = maxSize - veryPosSize

    veryNeg = addRows(veryNeg, veryNegAdd, veryNegSize)
    negative = addRows(negative, negativeAdd, negativeSize)
    neutral = addRows(neutral, neutralAdd, neutralSize)
    veryPos = addRows(veryPos, veryPosAdd, veryPosSize)

    frames = [veryNeg, negative, neutral, positive, veryPos]
    newCorpus = pd.concat(frames, ignore_index=True)
    resultCorpus = shuffle(newCorpus)
    return resultCorpus

def writeCSV(id, sentiment, fileName):
    solution = pd.DataFrame()
    solution.insert(0, "id", id)
    solution.insert(1, "sentiment", sentiment)
    solution.to_csv(fileName, index=False, sep=',')

def getDataSet(corpus):
    corpus = shuffle(corpus)

    veryNeg = corpus.loc[corpus['sentiment'] == 0]
    veryPos = corpus.loc[corpus['sentiment'] == 4]

    corpus = corpus.ix[corpus['sentiment'] != 0]
    corpus = corpus.ix[corpus['sentiment'] != 4]

    rowNumber = corpus.shape[0]
    splitPoint = int(rowNumber * percent)
    train = corpus.iloc[0:splitPoint]
    test = corpus.iloc[splitPoint:]

    negNumber = veryNeg.shape[0]
    negPoint = int(negNumber * percent)
    veryNeg = shuffle(veryNeg)
    X = veryNeg.iloc[0:negPoint]
    train = pd.concat([train.copy(), X.copy()], ignore_index=True)
    X = veryNeg.iloc[negPoint:]
    test = pd.concat([test.copy(), X.copy()], ignore_index=True)

    posNumber = veryPos.shape[0]
    posPoint = int(posNumber * percent)
    veryPos = shuffle(veryPos)
    X = veryPos.iloc[0:posPoint]
    train = pd.concat([train.copy(), X.copy()], ignore_index=True)
    X = veryPos.iloc[posPoint:]
    test = pd.concat([test.copy(), X.copy()], ignore_index=True)

    return train, test

if __name__ == '__main__':

    if len(argv) == 2:
        percent = float(argv[1])
    else:
        percent = 0.50

    corpus, observe = load_data()

    train, test = getDataSet(corpus)

    for n in range(500, 7000, 100):
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=n, token_pattern='(?u)\\b[a-zA-Z]\\w{2,}\\b', stop_words='english', min_df=1)
        # tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern='(?u)\\b[a-zA-Z]\\w{2,}\\b', stop_words='english', min_df=1)
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus["text"])

        # clf = MultinomialNB().fit(tfidf_matrix, train["sentiment"])
        clf = BernoulliNB().fit(tfidf_matrix, corpus["sentiment"])
        # clf = LogisticRegression().fit(tfidf_matrix, train["sentiment"])
        # clf = NuSVC().fit(tfidf_matrix, train["sentiment"])

        test_term_freq_matrix = tfidf_vectorizer.transform(test["text"])
        tested = clf.predict(test_term_freq_matrix)

        # for text, sentiment in zip(observe["text"], tested):
        #     print('%r => %s' % (text, sentiment))

        print("acc = %0.4f, max = %d" % (np.mean(tested == test["sentiment"]), n))

