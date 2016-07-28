import pickle
import datetime
import logging
import random
import string

import numpy as np
import pandas as pd

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

# --- Matthews Correlation Coefficient


def cov(x, y):
    N = np.shape(x)[0]
    K = np.shape(x)[1]
    x_centered = (x - np.mean(x, 0))
    y_centered = (y - np.mean(y, 0))
    cov = 0
    for n in range(0, N):
        cov += x_centered[n].T.dot(y_centered[n])
    return cov / K


def MCC(x, y):
    # check if the input is in multiclass form
    try:
        y_type, _, _ = sklearn.metrics.classification._check_targets(x, y)
        if y_type.startswith('multiclass'):
            x = lb.transform(x)
            y = lb.transform(y)
    except:
        print('Warning, input not sklearn compatible: ' +
              'No metrics support "multiclass-multioutput" format')

    return (cov(x, y) / np.sqrt(cov(x, x) * cov(y, y)))


def CV(estimator, parameters, features_train, features_test):

    mcc_train_all = []
    mcc_test_all = []
    f1_micro_train_all = []
    f1_micro_test_all = []
    f1_macro_train_all = []
    f1_macro_test_all = []

    for fold in range(0, num_folds):

        logger.info("# Fold", fold+1, "/", num_folds)

        grid_search_cv = sklearn.grid_search.GridSearchCV
        grid_search = grid_search_cv(estimator,
                                     parameters,
                                     scoring=sklearn.metrics.make_scorer(MCC),
                                     cv=5, n_jobs=CORES)
        grid_search.fit(self.data_splits[l_key].X_train,
                        self.data_splits[l_key].Y_train)

        estimator.fit(features_train[fold], folds_train_Y[fold])
        predictions_train = estimator.predict(features_train[fold])
        predictions_test = estimator.predict(features_test[fold])

        mcc_train = MCC(predictions_train, folds_train_Y[fold])
        mcc_test = MCC(predictions_test, folds_test_Y[fold])
        f1_micro_train = sklearn.metrics.f1_score(
            predictions_train, folds_train_Y[fold], average='micro')
        f1_micro_test = sklearn.metrics.f1_score(
            predictions_test, folds_test_Y[fold], average='micro')
        f1_macro_train = sklearn.metrics.f1_score(
            predictions_train, folds_train_Y[fold], average='macro')
        f1_macro_test = sklearn.metrics.f1_score(
            predictions_test, folds_test_Y[fold], average='macro')
        # print(float(np.round(mcc_train, 3)), "MCC train",)
        # print(float(np.round(mcc_test, 3)), "MCC test")
        # print(float(np.round(f1_micro_train, 3)), "F1 micro train")
        # print(float(np.round(f1_micro_test, 3)), "F1 micro test")
        # print(float(np.round(f1_macro_train, 3)), "F1 macro train")
        # print(float(np.round(f1_macro_test, 3)), "F1 macro test")

        mcc_train_all.append(mcc_train)
        mcc_test_all.append(mcc_test)
        f1_micro_train_all.append(f1_micro_train)
        f1_micro_test_all.append(f1_micro_test)
        f1_macro_train_all.append(f1_macro_train)
        f1_macro_test_all.append(f1_macro_test)

    print("# Average")
    print(float(np.round(np.mean(mcc_train_all), 3)), "+-",
          float(np.round(np.std(mcc_train_all), 3)), "MCC train",)
    print(float(np.round(np.mean(mcc_test_all), 3)), "+-",
          float(np.round(np.std(mcc_test_all), 3)), "MCC test")
    print(float(np.round(np.mean(f1_micro_train_all), 3)), "+-",
          float(np.round(np.std(f1_micro_train_all), 3)), "F1 micro train")
    print(float(np.round(np.mean(f1_micro_test_all), 3)), "+-",
          float(np.round(np.std(f1_micro_test_all), 3)), "F1 micro test")
    print(float(np.round(np.mean(f1_macro_train_all), 3)), "+-",
          float(np.round(np.std(f1_macro_train_all), 3)), "F1 macro train")
    print(float(np.round(np.mean(f1_macro_test_all), 3)), "+-",
          float(np.round(np.std(f1_macro_test_all), 3)), "F1 macro test")


def CV_all_feature_spaces(estimator):
    print("# -- N-grams")
    CV(estimator, features_ngrams_train, features_ngrams_test)
    print("# -- Bag-of-Means")
    CV(estimator, features_bom_train, features_bom_test)
    print("# -- PV-DBOW")
    CV(estimator, features_pvdbow_train, features_pvdbow_test)
    print("# -- PV-DM")
    CV(estimator, features_pvdm_train, features_pvdm_test)


# --- Basic setup

TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")
CORES = multiprocessing.cpu_count()

outfolder = "../output"
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

base_out_folder = outfolder + "/" + EXP_NAME
if not os.path.exists(base_out_folder):
    os.makedirs(base_out_folder)

logfolder = base_out_folder + "/" + "logs"
if not os.path.exists(logfolder):
    os.makedirs(logfolder)

resultfolder = base_out_folder + "/" + "results"
if not os.path.exists(resultfolder):
    os.makedirs(resultfolder)


# --- Logger setup

# create logger
logger = logging.getLogger(EXP_NAME)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(logfolder + '/logger.log')
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

lb = pickle.load(open("sentences_lb.pickle", "rb"))

folds_train_X = pickle.load(open("sentences_folds_train_X.pickle", "rb"))
folds_train_Y = pickle.load(open("sentences_folds_train_Y.pickle", "rb"))
folds_test_X = pickle.load(open("sentences_folds_test_X.pickle", "rb"))
folds_test_Y = pickle.load(open("sentences_folds_test_Y.pickle", "rb"))

features_ngrams_train = pickle.load(
    open("sentences_features_ngrams_train.pickle", "rb"))
features_ngrams_test = pickle.load(
    open("sentences_features_ngrams_test.pickle", "rb"))

features_bom_train = pickle.load(
    open("sentences_features_bom_train.pickle", "rb"))
features_bom_test = pickle.load(
    open("sentences_features_bom_test.pickle", "rb"))

features_pvdbow_train = pickle.load(
    open("sentences_features_pvdbow_train.pickle", "rb"))
features_pvdbow_test = pickle.load(
    open("sentences_features_pvdbow_test.pickle", "rb"))

features_pvdm_train = pickle.load(
    open("sentences_features_pvdm_train.pickle", "rb"))
features_pvdm_test = pickle.load(
    open("sentences_features_pvdm_test.pickle", "rb"))


# --- Run classifiers

scorer = sklearn.metrics.make_scorer(MCC)

# -- Logistic Regression

print("# --- Logistic Regression")

classifier = sklearn.linear_model.LogisticRegression()
estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
CV_all_feature_spaces(estimator)

# -- Decision Tree

print("# --- Decision Tree")

classifier = sklearn.tree.DecisionTreeClassifier()
estimator = sklearn.multiclass.OneVsRestClassifier(classifier)

# -- Naive Bayes

print("# --- Naive Bayes")

classifier = sklearn.naive_bayes.MultinomialNB()
estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
