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

import sklearn
import sklearn.cross_validation
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.tree
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.grid_search

import gensim


# --- Basic setup

TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")
VALIDATION_SIZE = 0.2
CORES = multiprocessing.cpu_count()

INPUTFOLDER = "../output/generate_best_feature_spaces/results/"

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


# --- Load data

num_folds = 5

lb = pickle.load(open(INPUTFOLDER + "/lb.pickle", "rb"))

folds_train_X = pickle.load(
    open(INPUTFOLDER + "/folds_train_X.pickle", "rb"))
folds_train_Y = pickle.load(
    open(INPUTFOLDER + "/folds_train_Y.pickle", "rb"))
folds_test_X = pickle.load(open(
    INPUTFOLDER + "/folds_test_X.pickle", "rb"))
folds_test_Y = pickle.load(open(
    INPUTFOLDER + "/folds_test_Y.pickle", "rb"))

features_ngrams_train = pickle.load(
    open(INPUTFOLDER + "/features_ngrams_train.pickle", "rb"))
features_ngrams_test = pickle.load(
    open(INPUTFOLDER + "/features_ngrams_test.pickle", "rb"))

features_bom_train = pickle.load(
    open(INPUTFOLDER + "/features_bom_train.pickle", "rb"))
features_bom_test = pickle.load(
    open(INPUTFOLDER + "/features_bom_test.pickle", "rb"))

features_pv_train = pickle.load(
    open(INPUTFOLDER + "/features_pv_train.pickle", "rb"))
features_pv_test = pickle.load(
    open(INPUTFOLDER + "/features_pv_test.pickle", "rb"))


# --- Functions

# -- Evaluation


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
            x = le.inverse_transform(x)
            y = le.inverse_transform(y)
            x = lb.transform(x)
            y = lb.transform(y)

    except:
        msg = ('Warning, input not sklearn compatible: ' +
               'No metrics support "multiclass-multioutput" format')
        logger.warning(msg)

    return (cov(x, y) / np.sqrt(cov(x, x) * cov(y, y)))


def CV(estimator, parameters, X_train, X_test, Y_train, Y_test):

    mcc_train_all = []
    mcc_test_all = []

    for fold in range(0, num_folds):

        logger.info("# - Fold " + str(fold+1) + " / " + str(num_folds))

        grid_search_cv = sklearn.grid_search.GridSearchCV
        grid_search = grid_search_cv(estimator,
                                     parameters,
                                     scoring=sklearn.metrics.make_scorer(MCC),
                                     cv=5, n_jobs=1)
        grid_search.fit(X_train[fold], Y_train[fold])

        predictions_train = grid_search.predict(X_train[fold])
        predictions_test = grid_search.predict(X_test[fold])

        mcc_train = MCC(predictions_train, Y_train[fold])
        mcc_test = MCC(predictions_test, Y_test[fold])
        mcc_train_all.append(mcc_train)
        mcc_test_all.append(mcc_test)

    logger.info("# Train: " + str(float(np.round(np.mean(mcc_train_all), 3))) +
                " +- " + str(float(np.round(np.std(mcc_train_all), 3))))
    logger.info("# Test:  " + str(float(np.round(np.mean(mcc_test_all), 3))) +
                " +- " + str(float(np.round(np.std(mcc_test_all), 3))))


def CV_all_feature_spaces(estimator, parameters, folds_train_Y, folds_test_Y):
    for i in range(len(features_names)):
        logger.info("# -- " + features_names[i])
        CV(estimator, parameters,
           features_train[i], features_test[i],
           folds_train_Y, folds_test_Y)


# --- prepare data

# --- prepare features

# features_names = ['N-Grams', 'Bag-of-Means', 'Paragraph Vectors']
# features_train = [features_ngrams_train, features_bom_train, features_pv_train]
# features_test = [features_ngrams_test, features_bom_test, features_pv_test]

features_names = ['N-Grams']
features_train = [features_ngrams_train]
features_test = [features_ngrams_test]

# -- prepare labels also as categorical variable for multinomial algorithms

folds_train_Y_cat = []
folds_test_Y_cat = []

le = sklearn.preprocessing.LabelEncoder()
le.fit(lb.classes_)

for fold in range(len(folds_train_Y)):
    folds_train_Y_cat_inv = lb.inverse_transform(folds_train_Y[fold])
    # folds_train_Y_cat.append(le.transform(folds_train_Y[fold]))
    folds_train_Y_cat.append(le.transform(folds_train_Y_cat_inv))
    folds_test_Y_cat_inv = lb.inverse_transform(folds_test_Y[fold])
    # folds_test_Y_cat.append(le.transform(folds_test_Y[fold]))
    folds_test_Y_cat.append(le.transform(folds_test_Y_cat_inv))


# --- Run classifiers

# -- Logistic Regression

logger.info("# --- Logistic Regression (one-vs-rest)")

classifier = sklearn.linear_model.LogisticRegression()
estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
parameter_space = {
    'estimator__C': [0.1, 1, 10, 100]
}
CV_all_feature_spaces(estimator, parameter_space,
                      folds_train_Y, folds_test_Y)

logger.info("# --- Logistic Regression (multinomial)")

classifier = sklearn.linear_model.LogisticRegression(
    multi_class="multinomial", solver='lbfgs')
parameter_space = {
    # 'C': [0.1, 1, 10, 100]
    }
CV_all_feature_spaces(classifier, parameter_space,
                      folds_train_Y_cat, folds_test_Y_cat)

# # -- Decision Tree
#
# logger.info("# --- Decision Tree")
#
# classifier = sklearn.tree.DecisionTreeClassifier()
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
# CV_all_feature_spaces(estimator)
#
# # -- Naive Bayes
#
# logger.info("# --- Naive Bayes")
#
# classifier = sklearn.naive_bayes.MultinomialNB()
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
#
# # -- Random Forest
#
# logger.info("# --- Random Forest")
#
# classifier = sklearn.ensemble.RandomForestClassifier()
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
