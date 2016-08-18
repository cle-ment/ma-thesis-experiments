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
import sklearn.pipeline
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

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Embedding, Lambda
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from keras.wrappers import scikit_learn
from keras import backend as K

# --- Basic setup

LOCAL = True

TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")
VALIDATION_SIZE = 0.2
CORES = multiprocessing.cpu_count()
FEAT_DIM = 300
OUTPUT_DIM = 6

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

NUM_FOLDS = 5

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

# feature transformation

class DenseTransformer(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()

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


def CV(estimator, parameters, X_train, X_test, Y_train, Y_test, n_jobs=CORES):

    for fold in range(0, NUM_FOLDS):

        if computation_progress['fold'] > fold:
            continue

        logger.info("# - Fold " + str(fold+1) + " / " + str(NUM_FOLDS))

        grid_search_cv = sklearn.grid_search.GridSearchCV
        grid_search = grid_search_cv(estimator,
                                     parameters,
                                     scoring=sklearn.metrics.make_scorer(MCC),
                                     cv=5, n_jobs=n_jobs)
        print("grid search start")
        grid_search.fit(X_train[fold], Y_train[fold])
        print("grid end")
        logger.info("# - Done. Best params: " + str(grid_search.best_params_))

        print("predict train start")
        predictions_train = grid_search.predict(X_train[fold])
        print("predict train end")
        print("predict test start")
        predictions_test = grid_search.predict(X_test[fold])
        print("predict test end")

        mcc_train = MCC(predictions_train, Y_train[fold])
        mcc_test = MCC(predictions_test, Y_test[fold])
        computation_progress['current_folds_train'].append(mcc_train)
        computation_progress['current_folds_test'].append(mcc_test)
        # update status
        computation_progress['fold'] = fold+1
        store_results(computation_progress)

    # store results
    mcc_train_all = computation_progress['current_folds_train']
    mcc_test_all = computation_progress['current_folds_test']

    logger.info("# -- MCC Train: " +
                str(float(np.round(np.mean(mcc_train_all), 3))) +
                " +- " + str(float(np.round(np.std(mcc_train_all), 3))))
    logger.info("# -- MCC Test:  " +
                str(float(np.round(np.mean(mcc_test_all), 3))) +
                " +- " + str(float(np.round(np.std(mcc_test_all), 3))))

    # reset status
    computation_progress['fold'] = 0
    computation_progress['current_folds_train'] = []
    computation_progress['current_folds_test'] = []
    store_results(computation_progress)

    return np.mean(mcc_train_all), np.mean(mcc_test_all)


def CV_all_feature_spaces(estimator, estimator_name, parameters,
                          folds_train_Y, folds_test_Y, n_jobs=CORES):

    for i in range(len(features_names)):

        if computation_progress['features'] > i:
            continue

        logger.info("# -- " + features_names[i])
        mcc_train, mcc_test = CV(estimator, parameters,
                                 features_train[i], features_test[i],
                                 folds_train_Y, folds_test_Y, n_jobs=n_jobs)

        computation_progress['results'].append(
            [features_names[i], estimator_name, mcc_train, mcc_test]
        )
        # store results in DataFrame
        results_df = pd.DataFrame(computation_progress['results'])
        results_df.columns = ['features', 'classifier',
                              'mcc train', 'mcc test']
        results_df.to_csv(RESULTFOLDER + "/results.csv")
        # update status
        computation_progress['features'] = i+1
        store_results(computation_progress)

    # when done reset status to 0
    computation_progress['features'] = 0
    store_results(computation_progress)


# -- Storing results to server in case of interuption

def store_results(computation_progress,
                  source="/home/ubuntu/thesis-experiments/output/",
                  target="clemens@cwestrup.de:thesis/output"):
    pickle.dump(computation_progress,
                open(BASE_OUTFOLDER + "/computation_progress.pickle", "wb"))
    if not LOCAL:
        call(["rsync", "-av", "--update", "--delete", "--force", source, target])


# --- prepare data

# -- prepare features

# for local experiments take only N-Grams

if LOCAL:
    features_names = ['N-Grams']
    features_train = [features_ngrams_train]
    features_test = [features_ngrams_test]
else:
    features_names = ['N-Grams', 'Bag-of-Means', 'Paragraph Vectors']
    features_train = [features_ngrams_train, features_bom_train,
                      features_pv_train]
    features_test = [features_ngrams_test, features_bom_test, features_pv_test]

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

# -- check where computation was left of in case of interuption
try:
    computation_progress = pickle.load(
        open(BASE_OUTFOLDER + "/computation_progress.pickle", "rb"))
except:
    # no computation was done yet so create a progress file_prefix
    computation_progress = {'classifier': 0,
                            'features': 0,
                            'fold': 0,
                            'results': [],
                            'current_folds_train': [],
                            'current_folds_test': [],
                            }
    store_results(computation_progress)

print(computation_progress['current_folds_train'])



# --- Run classifiers
#
# if (computation_progress['classifier'] == 0):
#
#     logger.info("# --- Logistic Regression (one-vs-rest)")
#
#     classifier = sklearn.linear_model.LogisticRegression()
#     estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
#     parameter_space = {
#         'estimator__C': [0.01, 0.1, 1, 10]
#     }
#     CV_all_feature_spaces(estimator, "Logistic Regression (one-vs-rest)",
#                           parameter_space, folds_train_Y, folds_test_Y)
#
#     # update status
#     computation_progress['classifier'] = 1
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 1):
#
#     logger.info("# --- Logistic Regression (multinomial)")
#
#     classifier = sklearn.linear_model.LogisticRegression(
#         multi_class="multinomial", solver='lbfgs')
#     parameter_space = {
#         'C': [0.1, 1, 10, 100]
#         }
#     CV_all_feature_spaces(classifier, "Logistic Regression (multinomial)",
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat)
#
#     # update status
#     computation_progress['classifier'] = 2
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 2):
#
#     logger.info("# --- Multinomial Naive Bayes (one-vs-rest)")
#
#     classifier = sklearn.naive_bayes.MultinomialNB()
#     estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
#     parameter_space = {
#         'estimator__alpha': [0.1, 1, 10, 100, 1000]
#     }
#     CV_all_feature_spaces(estimator, "Multinomial Naive Bayes (one-vs-rest)",
#                           parameter_space, folds_train_Y, folds_test_Y)
#
#     # update status
#     computation_progress['classifier'] = 3
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 3):
#
#     logger.info("# --- Multinomial Naive Bayes (multinomial)")
#
#     classifier = sklearn.naive_bayes.MultinomialNB()
#     parameter_space = {
#         'alpha': [0.1, 1, 10, 100, 1000]
#         }
#     CV_all_feature_spaces(classifier, "Multinomial Naive Bayes (multinomial)",
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat)
#
#     # update status
#     computation_progress['classifier'] = 4
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 4):
#
#     logger.info("# --- Decision Tree (one-vs-rest)")
#
#     classifier = sklearn.tree.DecisionTreeClassifier()
#     estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
#     parameter_space = {
#         'estimator__criterion': ['gini', 'entropy'],
#         'estimator__max_features': [None, 'auto', 'sqrt', 'log2']
#         }
#     CV_all_feature_spaces(estimator, "Decision Tree (one-vs-rest)",
#                           parameter_space, folds_train_Y, folds_test_Y)
#
#     # update status
#     computation_progress['classifier'] = 5
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 5):
#
#     logger.info("# --- Decision Tree (multinomial)")
#
#     classifier = sklearn.tree.DecisionTreeClassifier()
#     parameter_space = {
#         'criterion': ['gini', 'entropy'],
#         'max_features': [None, 'auto', 'sqrt', 'log2']
#         }
#     CV_all_feature_spaces(classifier, "Decision Tree (multinomial)",
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat)
#
#     # update status
#     computation_progress['classifier'] = 6
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 6):
#
#     logger.info("# --- Random Forest (one-vs-rest)")
#
#     classifier = sklearn.ensemble.RandomForestClassifier()
#     estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
#     parameter_space = {
#         'estimator__n_estimators': [5, 10, 20],
#         'estimator__criterion': ['gini', 'entropy'],
#         'estimator__max_features': [None, 'auto', 'sqrt', 'log2']
#         }
#     CV_all_feature_spaces(estimator, "Random Forest (one-vs-rest)",
#                           parameter_space, folds_train_Y, folds_test_Y)
#
#     # update status
#     computation_progress['classifier'] = 7
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 7):
#
#     logger.info("# --- Random Forest (multinomial)")
#
#     classifier = sklearn.ensemble.RandomForestClassifier()
#     parameter_space = {
#         'n_estimators': [5, 10, 20],
#         'criterion': ['gini', 'entropy'],
#         'max_features': [None, 'auto', 'sqrt', 'log2']
#         }
#     CV_all_feature_spaces(classifier, "Random Forest (multinomial)",
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat)
#
#     # update status
#     computation_progress['classifier'] = 8
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 8):
#
#     logger.info("# --- k Nearest Neighbors (one-vs-rest)")
#
#     classifier = sklearn.neighbors.KNeighborsClassifier()
#     estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
#     parameter_space = {
#         'estimator__n_neighbors': [3, 5, 10],
#         'estimator__weights': ['uniform', 'distance']
#         }
#     CV_all_feature_spaces(estimator, "k Nearest Neighbors (one-vs-rest)",
#                           parameter_space, folds_train_Y, folds_test_Y)
#
#     # update status
#     computation_progress['classifier'] = 9
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 9):
#
#     logger.info("# --- k Nearest Neighbors (multinomial)")
#
#     classifier = sklearn.neighbors.KNeighborsClassifier()
#     parameter_space = {
#         'n_neighbors': [3, 5, 10],
#         'weights': ['uniform', 'distance']
#         }
#     CV_all_feature_spaces(classifier, "k Nearest Neighbors (multinomial)",
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat)
#
#     # update status
#     computation_progress['classifier'] = 10
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 10):
#
#     logger.info("# --- SVM (one-vs-one)")
#
#     classifier = sklearn.svm.SVC()
#     parameter_space = {
#         'C': [0.1, 1, 5, 10],
#         'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
#         'degree': [2, 3, 4],
#         'gamma': ['auto', 0.1, 1, 10, 100]
#         }
#     CV_all_feature_spaces(classifier, "SVM (one-vs-one)",
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat)
#
#     # update status
#     computation_progress['classifier'] = 11
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 11):
#
#     logger.info("# --- SVM (multinomial)")
#
#     classifier = sklearn.svm.LinearSVC(multi_class='crammer_singer')
#     parameter_space = {
#         'C': [0.1, 1, 5, 10]
#         }
#     CV_all_feature_spaces(classifier, "SVM (multinomial)",
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat)
#
#     # update status
#     computation_progress['classifier'] = 12
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 12):
#
#     clf_name = "Simple Neural Network (hidden layer: 128)"
#
#     logger.info("# --- " + clf_name)
#
#     def nn_model():
#         model = keras.models.Sequential()
#
#         model.add(Dense(128, input_dim=FEAT_DIM, init='uniform'))
#         model.add(Activation('tanh'))
#         model.add(Dropout(0.4))
#         model.add(Dense(OUTPUT_DIM, init='uniform'))
#         model.add(Activation('softmax'))
#
#         sgd = SGD(lr=0.05, decay=1e-6, momentum=0.7, nesterov=True)
#         model.compile(loss='categorical_crossentropy', optimizer=sgd)
#
#         return model
#
#     classifier = keras.wrappers.scikit_learn.KerasClassifier(
#         build_fn=nn_model,
#         nb_epoch=100)
#     scaler = sklearn.preprocessing.StandardScaler()
#     dense = DenseTransformer()
#
#     pipeline = sklearn.pipeline.Pipeline(
#         [('dense', dense),
#          ('scaler', scaler),
#          ('nn', classifier)])
#
#     parameter_space = {
#     # TODO: find out how to use paramters
#         # 'nn__model__lr': [0.01, 0.05, 0.1]
#     }
#
#
#     CV_all_feature_spaces(pipeline, clf_name,
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat,
#                           n_jobs=1)
#
#     # update status
#     computation_progress['classifier'] = 13
#     store_results(computation_progress)
#
# if (computation_progress['classifier'] == 13):
#
#     clf_name = "Deep Neural Network (hidden layers: 128 x 64 x 32)"
#
#     logger.info("# --- " + clf_name)
#
#     def nn_model():
#         model = keras.models.Sequential()
#
#         model.add(Dense(128, input_dim=FEAT_DIM, init='uniform'))
#         model.add(Activation('tanh'))
#         model.add(Dropout(0.3))
#         model.add(Dense(64, init='uniform'))
#         model.add(Activation('tanh'))
#         model.add(Dropout(0.3))
#         model.add(Dense(32, init='uniform'))
#         model.add(Activation('tanh'))
#         model.add(Dropout(0.3))
#         model.add(Dense(OUTPUT_DIM, init='uniform'))
#         model.add(Activation('softmax'))
#
#         sgd = SGD(lr=0.05, decay=1e-6, momentum=0.7, nesterov=True)
#         model.compile(loss='categorical_crossentropy', optimizer=sgd)
#
#         return model
#
#     classifier = keras.wrappers.scikit_learn.KerasClassifier(
#         build_fn=nn_model,
#         nb_epoch=100)
#     scaler = sklearn.preprocessing.StandardScaler()
#     dense = DenseTransformer()
#
#     pipeline = sklearn.pipeline.Pipeline(
#         [('dense', dense),
#          ('scaler', scaler),
#          ('nn', classifier)])
#
#     parameter_space = {
#     # TODO: find out how to use paramters
#         # 'nn__model__lr': [0.01, 0.05, 0.1]
#     }
#
#     CV_all_feature_spaces(pipeline, clf_name,
#                           parameter_space, folds_train_Y_cat, folds_test_Y_cat,
#                           n_jobs=1)
#
#     # update status
#     computation_progress['classifier'] = 14
#     store_results(computation_progress)


# TODO: remove
computation_progress['classifier'] = 14


if (computation_progress['classifier'] == 14):

    clf_name = "Convolutional Neural Network)"

    logger.info("# --- " + clf_name)

    # we use max over time pooling by defining a python function to use
    # in a Lambda layer
    def max_1d(X):
        return K.max(X, axis=1)

    def nn_model():

        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(FEAT_DIM,
                            100,
                            dropout=0.2))

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        model.add(Convolution1D(nb_filter=32,
                                filter_length=5,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        # max pooling
        model.add(Lambda(max_1d, output_shape=(32,)))
        # We add a vanilla hidden layer:
        model.add(Dense(64))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        # We project onto a single unit output layer,
        # and squash it with a sigmoid:
        model.add(Dense(OUTPUT_DIM))
        model.add(Activation('sigmoid'))

        sgd = SGD(lr=0.03, decay=1e-6, momentum=0.3, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)

        return model

    classifier = keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=nn_model,
        nb_epoch=100)
    scaler = sklearn.preprocessing.StandardScaler()
    dense = DenseTransformer()

    pipeline = sklearn.pipeline.Pipeline(
        [('dense', dense),
         ('scaler', scaler),
         ('nn', classifier)])

    parameter_space = {
    # TODO: find out how to use paramters
        # 'nn__model__lr': [0.01, 0.05, 0.1]
    }

    CV_all_feature_spaces(pipeline, clf_name,
                          parameter_space, folds_train_Y_cat, folds_test_Y_cat,
                          n_jobs=1)

    # update status
    computation_progress['classifier'] = 15
    store_results(computation_progress)

logger.info("Done.")
