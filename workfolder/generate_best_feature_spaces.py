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
import sys

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

# check if results are already there

if (os.path.exists(RESULTFOLDER + "/lb.pickle") and
    os.path.exists(RESULTFOLDER + "/folds_train_X.pickle") and
    os.path.exists(RESULTFOLDER + "/folds_train_Y.pickle") and
    os.path.exists(RESULTFOLDER + "/folds_test_X.pickle") and
    os.path.exists(RESULTFOLDER + "/folds_test_Y.pickle") and
    os.path.exists(RESULTFOLDER + "/features_ngrams_train.pickle") and
    os.path.exists(RESULTFOLDER + "/features_ngrams_test.pickle") and
    os.path.exists(RESULTFOLDER + "/features_bom_train.pickle") and
    os.path.exists(RESULTFOLDER + "/features_bom_test.pickle") and
    os.path.exists(RESULTFOLDER + "/features_pv_train.pickle") and
        os.path.exists(RESULTFOLDER + "/features_pv_test.pickle")):

    logger.info("Feature spaces already generated, exiting.")
    sys.exit()


# --- Logger setup

# create logger
logger = logging.getLogger(EXP_NAME)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(LOGFOLDER + '/logger.log')
fh.setLevel(logging.DEBUG)

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
pickle.dump(lb, open(RESULTFOLDER + "/lb.pickle", "wb"))

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
            open(RESULTFOLDER + "/folds_train_X.pickle", "wb"))
pickle.dump(folds_train_Y,
            open(RESULTFOLDER + "/folds_train_Y.pickle", "wb"))
pickle.dump(folds_test_X,
            open(RESULTFOLDER + "/folds_test_X.pickle", "wb"))
pickle.dump(folds_test_Y,
            open(RESULTFOLDER + "/folds_test_Y.pickle", "wb"))

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
            open(RESULTFOLDER +
                 "/features_ngrams_test.pickle", "wb"))
pickle.dump(features_ngrams_train,
            open(RESULTFOLDER +
                 "/features_ngrams_train.pickle", "wb"))

logger.info("Done")


# -- Bag-of-Means Word2Vec

logger.info("Generating Bag-of-Means Word2Vec")

features_bom_train = []
features_bom_test = []

model_word2vec = gensim.models.Word2Vec.load_word2vec_format(
    "./data/GoogleNews-vectors-negative300.bin.gz",
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
            open(RESULTFOLDER + "/features_bom_train.pickle", "wb"))
pickle.dump(features_bom_test,
            open(RESULTFOLDER + "/features_bom_test.pickle", "wb"))

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
    model, corpus_train, corpus_test, y_train,
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
    best_features_train = None
    best_features_test = None
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

        estimator_logreg = sklearn.multiclass.OneVsRestClassifier(
            sklearn.linear_model.LogisticRegression())

        # calculate scores
        mcc_train = score(estimator_logreg, features_train, y_train)

        # if a better score was achieved update the best model
        if mcc_train > best_train_score:
            best_train_score = mcc_train
            best_features_train = features_train
            best_features_test = infer_vectors(
                model, corpus_test,
                infer_steps, infer_min_alpha, infer_alpha)

        # elapsed time
        now_time = time.time()
        time_elapsed = now_time-start_time
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        logger.debug("Step " + str(step + 1) + " / " + str(steps) +
                     " MCC train score: " + "{:0.3f}".format(mcc_train))

    return best_features_train, best_features_test


logger.info("Generating Paragraph Vectors")

features_pv_train = []
features_pv_test = []

# settings
# STEPS = 10
# INFER_STEPS = 10
# INFER_ALPHA = 0.1
# INFER_MIN_ALPHA = 0.0001
INFER_STEPS = 2000
INFER_ALPHA = 0.2
INFER_MIN_ALPHA = 0.002

# used classifier: logistic regression
estimator_logreg = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression())

for i in range(0, num_folds):

    logger.debug("Fold " + str(i+1) + " / " + str(num_folds))

    # prepare data as gensim corpera
    corpus_train = list(read_corpus(folds_train_X[i]))
    corpus_test = list(read_corpus(folds_test_X[i], tokens_only=True))

    # specify model: best from eval_doc2vec results
    model = gensim.models.Doc2Vec(
        window=8, negative=10, min_count=2, hs=1, sample=0, dm=0,
        workers=CORES, iter=10)
    model.build_vocab(corpus_train)

    # train model and take features with highest score
    features_train, features_test = train_eval_doc2vec(
        model, corpus_train, corpus_test, folds_train_Y[i],
        # estimator=estimator_logreg,
        steps=STEPS, infer_steps=INFER_STEPS,
        infer_alpha=INFER_ALPHA, infer_min_alpha=INFER_MIN_ALPHA)

    features_pv_train.append(features_train)
    features_pv_test.append(features_test)

# store
pickle.dump(features_pv_train,
            open(RESULTFOLDER + "/features_pv_train.pickle", "wb"))
pickle.dump(features_pv_test,
            open(RESULTFOLDER + "/features_pv_test.pickle", "wb"))

logger.info("Done")

logger.info("Syncing results to server")

call(["rsync", "-av", "--update", "--delete", "--force",
      RESULTFOLDER,
      "clemens@cwestrup.de:thesis/output/" + EXP_NAME + "/results"])

logger.info("Done.")
