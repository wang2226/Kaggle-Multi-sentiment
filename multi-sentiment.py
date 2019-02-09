import pandas as pd
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

def load_data():
    train = pd.read_csv('./Headline_Trainingdata.csv')
    test = pd.read_csv('./Headline_Testingdata.csv')

    return train, test

# extraction features from corpus

def bag_of_words(words):
    return dict([(word, True) for word in words])

def bigram_words(words, score_fn=BigramAssocMeasures.pmi, n=121):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)

def bag_of_words_not_in_set(words, badwords):
    return list(set(words) - set(badwords))

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

def bag_of_words_in_set(words, goodwords):
    return bigram_words(list(set(words) & set(goodwords)))

# dimensionality reduction
def preProcess(sentence):
    sentence = sentence.lower()
    # pattern = r'\b[a-zA-Z]\w{2,}\b'
    # pattern = r'\w+'

    regex = re.compile('^.$|\$?\d+(?:\.\d+)?%?|\.\.\.')
    # regex = re.compile('^.$|\.\.\.')
    toRemoved = regex.findall(sentence)

    pattern = r"""(?x)
                     (?:[A-Z]\.)+
                     |\$?\d+(?:\.\d+)?%?
                     |\w+(?:[-']\w+)*
                     |\.\.\.
                     |(?:[.,;"'?():-_`])
                  """

    tokenizer = RegexpTokenizer(pattern)
    tokens = tokenizer.tokenize(sentence)

    stopfile = 'english'
    badwords = stopwords.words(stopfile)

    for word in toRemoved:
        badwords.append(word)

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    for punc in english_punctuations:
        badwords.append(punc)

    words = list(set(tokens) - set(badwords))

    stemmer = nltk.stem.PorterStemmer()
    stem = [stemmer.stem(t) for t in words]
    words = bag_of_non_stopwords(stem)

    return words

def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=0):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for label, sentences in labelled_words:
        for sent in sentences:
            words = preProcess(sent)
            for word in words:
                word_fd[word] += 1
                label_word_fd[label][word] += 1

    n_xx = label_word_fd.N()
    high_info_words = set()

    labelScore = []
    for label in sorted(label_word_fd.conditions()):
        # if label == 0:
        #     min_score = 1.0
        # elif label == 1:
        #     min_score = 1.0
        # elif label == 2:
        #     min_score = 1.0
        # elif label == 3:
        #     min_score = 1.0
        # elif label == 4:
        #     min_score = 1.0

        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)

        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score

        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)
        labelScore.append(word_scores)

    which = 0
    for x in labelScore:
        sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
        labelCSV = pd.DataFrame(sorted_x)
        fileName = "wang2226_%d.csv" % which
        labelCSV.to_csv(fileName, index=False, sep=',')
        which += 1

    return high_info_words


# selection features
def label_feats_from_testSet(test, feature_detector=bag_of_words):
    label_feats = []

    for sentence in test["text"]:
        words = preProcess(sentence)
        feats = feature_detector(words)
        if not feats:
            print("for pridict sentence=%s" % (sentence))
        label_feats.append(feats)
    return label_feats

def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    train_feats = []
    label_feats = collections.defaultdict(list)
    for label in corp["sentiment"].unique():
        for sentence in corp.loc[corp["sentiment"] == label].text:
            words = preProcess(sentence)
            feats = feature_detector(words)
            if not feats:
                print("for train label=%d, sentence=%s" % (label, sentence))
            label_feats[label].append(feats)

    for label, feats in label_feats.items():
        train_feats.extend([(feat, label) for feat in feats[:]])
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

def writeCSV(id, sentiment):
    solution = pd.DataFrame()
    solution.insert(0, "id", id)
    solution.insert(1, "sentiment", sentiment)
    solution.to_csv("wang2226_solu.csv", index=False, sep=',')

def get_feats(corpus, trainCorpus, testCorpus, pridictCorpus):
    labels = sorted(corpus["sentiment"].unique())
    labeled_words = [(l, corpus.loc[corpus["sentiment"] == l].text.tolist()) for l in labels]
    high_info_words = set(high_information_words(labeled_words))
    feat_det = lambda words: bag_of_words_in_set(words, high_info_words)
    trainFeats = label_feats_from_corpus(trainCorpus, feature_detector=feat_det)
    testFeats = label_feats_from_corpus(testCorpus, feature_detector=feat_det)
    pridictFeats = label_feats_from_testSet(pridictCorpus, feature_detector=feat_det)

    return trainFeats, testFeats, pridictFeats

def training_pool(trainFeats, testFeats, pridictFeats):
    maxAccuracy = 0.0
    bestPridict = pd.DataFrame()
    clfName = ""

    # # build classifier using NLTK Naive Bayes
    # # Training with Naive Bayes
    # classifier = NaiveBayesClassifier.train(trainFeats)
    # accuracy = nltk.classify.util.accuracy(classifier, testFeats)
    # # classifier.show_most_informative_features(20)
    # observed = classifier.classify_many(pridictFeats)
    #
    # if accuracy > maxAccuracy:
    #     maxAccuracy = accuracy
    #     bestPridict = observed
    #     clfName = "Naive Bayes"
    #
    # # Training with Naive Bayes with Laplace smoothing
    # classifier = NaiveBayesClassifier.train(trainFeats, estimator=LaplaceProbDist)
    # accuracy = nltk.classify.util.accuracy(classifier, testFeats)
    # # classifier.show_most_informative_features(20)
    # observed = classifier.classify_many(pridictFeats)
    #
    # if accuracy > maxAccuracy:
    #     maxAccuracy = accuracy
    #     bestPridict = observed
    #     clfName = "Naive Bayes with Laplace"
    #
    # # build classifier using SKLEARN
    # # Training with Multinomial Naive Bayes
    # classifier = SklearnClassifier(MultinomialNB())
    # classifier._vectorizer.sort = False
    # classifier.train(trainFeats)
    # accuracy = nltk.classify.util.accuracy(classifier, testFeats)
    # observed = classifier.classify_many(pridictFeats)
    #
    # if accuracy > maxAccuracy:
    #     maxAccuracy = accuracy
    #     bestPridict = observed
    #     clfName = "Multinomial Naive Bayes"

    # Training with Bernoulli Naive Bayes
    classifier = SklearnClassifier(BernoulliNB())
    classifier._vectorizer.sort = False
    classifier.train(trainFeats)
    accuracy = nltk.classify.util.accuracy(classifier, testFeats)
    observed = classifier.classify_many(pridictFeats)

    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        bestPridict = observed
        clfName = "Bernoulli Naive Bayes"

    # Training with logistic regression
    classifier = SklearnClassifier(LogisticRegression())
    classifier._vectorizer.sort = False
    classifier.train(trainFeats)
    accuracy = nltk.classify.util.accuracy(classifier, testFeats)
    observed = classifier.classify_many(pridictFeats)

    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        bestPridict = observed
        clfName = "Logistic Regression"

    # # Training with LinearSVC
    # classifier = SklearnClassifier(SVC())
    # classifier._vectorizer.sort = False
    # classifier.train(trainFeats)
    # accuracy = nltk.classify.util.accuracy(classifier, testFeats)
    # observed = classifier.classify_many(pridictFeats)
    #
    # if accuracy > maxAccuracy:
    #     maxAccuracy = accuracy
    #     bestPridict = observed
    #     clfName = "Linear SVC"

    # Training with NuSVC
    # classifier = SklearnClassifier(NuSVC())
    # classifier._vectorizer.sort = False
    # classifier.train(trainFeats)
    # accuracy = nltk.classify.util.accuracy(classifier, testFeats)
    # observed = classifier.classify_many(pridictFeats)
    #
    # if accuracy > maxAccuracy:
    #     maxAccuracy = accuracy
    #     bestPridict = observed
    #     clfName = "Nu SVC"

    return maxAccuracy, bestPridict, clfName

def balanceCorpus(corpus):
    veryNeg = corpus.loc[corpus['sentiment'] == 0]
    veryPos = corpus.loc[corpus['sentiment'] == 4]

    corpus = corpus.ix[corpus['sentiment'] != 0]
    corpus = corpus.ix[corpus['sentiment'] != 4]

    totalNumber = corpus.shape[0]
    splitPoint = int(totalNumber * percent)
    corpus = shuffle(corpus)
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
    train = shuffle(pd.concat([train, X.copy()], ignore_index=True))
    X = veryPos.iloc[posPoint:]
    test = shuffle(pd.concat([test, X.copy()], ignore_index=True))

    return train, test

if __name__ == '__main__':

    if len(argv) == 3:
        percent = float(argv[1])
    else:
        percent = 0.00

    corpus, pridict = load_data()

    maxAccuracy = 0.0
    bestPridict = []
    clfName = ""
    for x in range(0, 30):

        train, test = balanceCorpus(corpus)

        # balance corpus, add rows to veryNeg, negative, neutral, veryPos
        # rebuildCorpus = rebuild_corpus(corpus)

        # get the seed corpus
        trainFeats, testFeats, pridictFeats = get_feats(corpus, corpus, test, pridict)

        accuracy, pridictLabel, name = training_pool(trainFeats, testFeats, pridictFeats)

        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            bestPridict = pridictLabel
            clfName = name

    print('for best seed is %s, accuracy: %0.4f' % (clfName, maxAccuracy))
    writeCSV(pridict["id"], bestPridict)


