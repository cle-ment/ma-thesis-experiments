# --- Imports

import datetime
import logging
import pickle
import random
import time
import copy
import itertools
from itertools import compress
import os
import multiprocessing

from subprocess import call

import numpy as np
import pandas as pd

import gensim

import sklearn
import sklearn.cross_validation
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.tree
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.ensemble
import sklearn.preprocessing


# --- Basic setup

TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")
VALIDATION_SIZE = 0.2
CORES = multiprocessing.cpu_count()

OUTFOLDER = "../output"
if not os.path.exists(OUTFOLDER):
    os.makedirs(OUTFOLDER)

BASE_OUTFOLDER = OUTFOLDER + "/" + EXP_NAME
if not os.path.exists(BASE_OUTFOLDER):
    os.makedirs(BASE_OUTFOLDER)

LOGFOLDER = BASE_OUTFOLDER + "/" + "logs"
if not os.path.exists(LOGFOLDER):
    os.makedirs(LOGFOLDER)

RESULTFOLDER = BASE_OUTFOLDER + "/" + "results"
if not os.path.exists(RESULTFOLDER):
    os.makedirs(RESULTFOLDER)


# --- Logger setup

# create logger
logger = logging.getLogger(EXP_NAME)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(LOGFOLDER + '/logger.log')
fh.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s | %(message)s')

# add formatter to ch
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

logger.propagate = False


# --- Dataset Initialization

# read data
df_conf = pickle.load(open("data/sentences-dataframe-" +
                           "confidence_greater-0.6.pickle", "rb"))

# binarize labels
label_array = np.array(df_conf['0_label'])
lb = sklearn.preprocessing.LabelBinarizer()
lb.fit(label_array)
# store them
pickle.dump(lb, open(BASE_OUTFOLDER + "/sentences_lb.pickle", "wb"))

data_Y = lb.transform(label_array)
data_X = np.array(df_conf['0-sentence'])

# -- Split data into 5 folds
num_folds = 5
fold_len = int(np.ceil(len(data_X) / num_folds))

folds_train_X = []
folds_train_Y = []
folds_test_X = []
folds_test_Y = []

for i in range(0, num_folds):
    folds_test_X.append(data_X[i*fold_len:(i+1)*fold_len])
    folds_test_Y.append(data_Y[i*fold_len:(i+1)*fold_len])

    train_x = []
    train_y = []
    for j in range(0, num_folds):
        if j != i:
            train_x.append(data_X[j*fold_len:(j+1)*fold_len])
            train_y.append(data_Y[j*fold_len:(j+1)*fold_len])
    folds_train_X.append(np.concatenate(train_x))
    folds_train_Y.append(np.vstack(train_y))

pickle.dump(folds_train_X,
            open(BASE_OUTFOLDER + "/sentences_folds_train_X.pickle", "wb"))
pickle.dump(folds_train_Y,
            open(BASE_OUTFOLDER + "/sentences_folds_train_Y.pickle", "wb"))
pickle.dump(folds_test_X,
            open(BASE_OUTFOLDER + "/sentences_folds_test_X.pickle", "wb"))
pickle.dump(folds_test_Y,
            open(BASE_OUTFOLDER + "/sentences_folds_test_Y.pickle", "wb"))

# just to make sure we don't even try to use them,
# kill the test set labels :D
folds_test_Y = None


# --- Generate Feature Spaces

# -- N-grams

logger.info("Generating N-Grams")

features_ngrams_train = []
features_ngrams_test = []

for i in range(0, num_folds):
    # use best settings from previous experiments
    tfidf_vect = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 1),
        norm=None,
        stop_words=None,
        sublinear_tf=True,
        use_idf=True)
    # train ngrams model on training data
    features_ngrams_train.append(tfidf_vect.fit_transform(folds_train_X[i]))
    # only apply it on test data
    features_ngrams_test.append(tfidf_vect.transform(folds_test_X[i]))

pickle.dump(features_ngrams_test,
            open("sentences_features_ngrams_test.pickle", "wb"))
pickle.dump(features_ngrams_train,
            open("sentences_features_ngrams_train.pickle", "wb"))

logger.info("Done")


# -- Bag-of-Means Word2Vec

logger.info("Generating Bag-of-Means Word2Vec")

features_bom_train = []
features_bom_test = []

model_word2vec = gensim.models.Word2Vec.load_word2vec_format(
    "/Users/cwestrup/thesis/data/word2vec/GoogleNews-vectors-negative300.bin",
    binary=True)


def build_bagofmeans(X, dimensionality=300):
    means = []
    for sentence in X:
        words = sentence.split()
        word_vectors = []
        for word in words:
            try:
                word_vectors.append(model_word2vec[word])
            except:
                pass  # word not in vocab
        if word_vectors != []:
            mean = np.matrix(np.mean(word_vectors, axis=0))
        else:
            mean = np.zeros((1, dimensionality))
        means.append(mean)
    return np.vstack(means)

for i in range(0, num_folds):
    features_bom_train.append(build_bagofmeans(folds_train_X[i]))
    features_bom_test.append(build_bagofmeans(folds_test_X[i]))

pickle.dump(features_bom_train,
            open("sentences_features_bom_train.pickle", "wb"))
pickle.dump(features_bom_test,
            open("sentences_features_bom_test.pickle", "wb"))

logger.info("Done")


# -- Paragraph Vectors (Doc2Vec)


def cov(x, y):
    """ Covariance Function for Matthews Correlation coefficient below """

    N = np.shape(x)[0]
    K = np.shape(x)[1]
    x_centered = (x - np.mean(x, 0))
    y_centered = (y - np.mean(y, 0))
    cov = 0
    for n in range(0, N):
        cov += x_centered[n].T.dot(y_centered[n])
    return cov / K


def MCC(x, y):
    """ Matthews Correlation coefficient for K classes """

    # check if the input is in multiclass form
    try:
        y_type, _, _ = sklearn.metrics.classification._check_targets(x, y)
        if y_type.startswith('multiclass'):
            x = lb.transform(x)
            y = lb.transform(y)
    except:
        msg = ('Warning, input not sklearn compatible: ' +
               'No metrics support "multiclass-multioutput" format')
        logger.warning(msg)

    return (cov(x, y) / np.sqrt(cov(x, x) * cov(y, y)))


def score(estimator, X_train, Y_train):

    """ Evaluate the performance on given test and training data using the
        estimator (classifier) provided
    """

    estimator.fit(X_train, Y_train)
    predictions_train = estimator.predict(X_train)
    mcc_train = MCC(predictions_train, Y_train)

    return mcc_train


def read_corpus(documents, tokens_only=False):
    """ Read a list of documents and produce a document corpus readable
        by gensim's Doc2Vec class
    """
    for i, sentence in enumerate(documents):
        if tokens_only:
            yield gensim.utils.simple_preprocess(sentence)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(
                gensim.utils.simple_preprocess(sentence), [i])


def infer_vectors(model, corpus, steps, min_alpha, alpha):
    """ Infer document vectors given a trained model
    """
    inferred_vectors = []
    for doc in corpus:
        if type(doc) == list:
            inferred_vectors.append(model.infer_vector(
                doc, alpha=alpha, min_alpha=min_alpha, steps=steps))
        else:
            inferred_vectors.append(model.infer_vector(
                doc.words, alpha=alpha, min_alpha=min_alpha, steps=steps))
    return np.vstack(inferred_vectors)


def train_eval_doc2vec(
    model, corpus_train, y_train,
    estimator=sklearn.linear_model.LinearRegression(),
    steps=100, alpha_start=0.025, alpha_end=0.0001,
        infer_steps=5, infer_min_alpha=0.0001, infer_alpha=0.1):

    """ Train the given doc2vec model and and evaluate it at each step.
        Measures training performance which correlates with test performance as
        empirically shown in previous experiments (for certain model choices)
    """

    # measure start time
    start_time = time.time()

    # set learning rate
    alpha = alpha_start
    alpha_delta = (alpha - alpha_end) / steps

    # store and return the best model
    best_model = None
    best_train_score = 0

    # copy training corpus for evaluation
    corpus_train_fixed = copy.deepcopy(corpus_train)

    for step in range(steps):

        # shuffling gets best results
        random.shuffle(corpus_train)

        # train language model
        model.alpha, model.min_alpha = alpha, alpha
        model.train(corpus_train)

        # decrease learning rate
        if (alpha_end):
            alpha -= alpha_delta

        # inferred training vectors
        features_train = infer_vectors(
            model, corpus_train_fixed,
            infer_steps, infer_min_alpha, infer_alpha)
        # calculate scores
        mcc_train = score(estimator, features_train, y_train)

        # if a better score was achieved update the best model
        if mcc_train > best_train_score:
            best_model = copy.deepcopy(model)

        # elapsed time
        now_time = time.time()
        time_elapsed = now_time-start_time
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print("Step " + str(step + 1) + " / " + str(steps) +
               " MCC train score: " + "{:0.3f}".format(mcc_train))

    return best_model


logger.info("Generating Paragraph Vectors")

features_pv_train = []
features_pv_test = []

# settings
STEPS = 50
INFER_STEPS = 10

# used classifier: logistic regression
estimator_logreg = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression())

for i in range(0, num_folds):

    print("Fold", i+1, "/", num_folds)

    # prepare data as gensim corpera
    corpus_train = list(read_corpus(folds_train_X[i]))
    corpus_test = list(read_corpus(folds_test_X[i], tokens_only=True))

    # specify model
    model = gensim.models.Doc2Vec(
        dm=0, size=10, window=5, negative=2, min_count=2,
        hs=0, sample=1e-4, workers=CORES, iter=10)
    model.build_vocab(corpus_train)

    # train model (and take the one with the highest test score)
    best_model = train_eval_doc2vec(
        model, corpus_train, folds_train_Y[i],
        estimator=estimator_logreg, steps=STEPS, infer_steps=INFER_STEPS)

    # infer training and test vectors
    features_train = infer_vectors(
        best_model, corpus_train, INFER_STEPS, 0.0001, 0.1)
    features_test = infer_vectors(
        best_model, corpus_test, INFER_STEPS, 0.0001, 0.1)

    features_pv_train.append(features_train)
    features_pv_test.append(features_test)

# store
pickle.dump(features_pv_train,
            open("sentences_features_pv_train.pickle", "wb"))
pickle.dump(features_pv_test,
            open("sentences_features_pv_test.pickle", "wb"))

logger.info("Done")
