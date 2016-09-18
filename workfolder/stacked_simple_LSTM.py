from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

import sklearn
import sklearn.metrics

from optparse import OptionParser
import logging
import pickle
import sys
import numpy as np
from copy import deepcopy
import os
from shutil import rmtree

parser = OptionParser()
parser.add_option("-v", "--verbose", dest="verbose",  action="store_true",
                  default=False, help="Show debug output.")
parser.add_option("-t", "--testrun", dest="testrun",  action="store_true",
                  default=False, help="Only do a test run with some data.")

(options, args) = parser.parse_args()

MAX_LENGTH = 200
ITERATIONS = 50
FOLDS_LIMIT = 1
VALIDATION_SET_SIZE = 0.1

PADDING_TOKEN = "_"
LABELS = ["other",
          "candidate",
          "nextsteps",
          "job",
          "benefits",
          "company"]
num_labels = len(LABELS)

EXP_NAME = __file__.rstrip(".py")
OUTFOLDER = "../output"
BASE_OUTFOLDER = OUTFOLDER + "/" + EXP_NAME
LOGFOLDER = BASE_OUTFOLDER + "/" + "logs"
RESULTFOLDER = BASE_OUTFOLDER + "/" + "results"
CHECKPOINTFOLDER = RESULTFOLDER + "/" + "model-checkpoints"
INPUTFOLDER = "../output/generate_best_feature_spaces/results/"

# If testrun remove previous results
if options.testrun and os.path.exists(BASE_OUTFOLDER):
    # remove previous results if they exist
    rmtree(BASE_OUTFOLDER)

# -- Setup folder structure

if not os.path.exists(OUTFOLDER):
    os.makedirs(OUTFOLDER)
if not os.path.exists(BASE_OUTFOLDER):
    os.makedirs(BASE_OUTFOLDER)
if not os.path.exists(LOGFOLDER):
    os.makedirs(LOGFOLDER)
if not os.path.exists(RESULTFOLDER):
    os.makedirs(RESULTFOLDER)
if not os.path.exists(CHECKPOINTFOLDER):
    os.makedirs(CHECKPOINTFOLDER)


# create logger
logger = logging.getLogger(EXP_NAME)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOGFOLDER + '/logger.log')
fh.setLevel(logging.INFO)
if options.verbose:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s | %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
logger.propagate = False


# Evaluation functions
def cov(x, y):
    N = np.shape(x)[0]
    K = np.shape(x)[1]
    x_centered = (x - np.mean(x, 0))
    y_centered = (y - np.mean(y, 0))
    cov = []
    for n in range(0, N):
        cov.append(x_centered[n].T.dot(y_centered[n]))
    return np.sum(cov) / K


def mcc(x, y):
    mcc = (cov(x, y) / np.sqrt(cov(x, x) * cov(y, y)))
    return mcc


# Load data
input_folds_train_X = pickle.load(
    open(INPUTFOLDER + "/folds_train_X.pickle", "rb"))
input_folds_train_Y = pickle.load(
    open(INPUTFOLDER + "/folds_train_Y.pickle", "rb"))
input_folds_test_X = pickle.load(open(
    INPUTFOLDER + "/folds_test_X.pickle", "rb"))
input_folds_test_Y = pickle.load(open(
    INPUTFOLDER + "/folds_test_Y.pickle", "rb"))

# extract all chars
all_folds_X = deepcopy(input_folds_train_X)
all_folds_X.extend(input_folds_test_X)
all_text = PADDING_TOKEN
max_len = 0
for fold_X in all_folds_X:
    for sentence in fold_X:
        all_text += sentence
        if len(sentence) > max_len:
            max_len = len(sentence)
del all_folds_X
max_len = min(max_len, MAX_LENGTH)

logger.info("Max sequence lenght (max_len) is " + str(max_len))

# count chars and labels
chars = set(all_text)
num_chars = len(chars)
logger.info('Total labels: ' + str(num_labels))
logger.info('Total chars: ' + str(num_chars))

# index chars for X
chars2indices = dict((c, i) for i, c in enumerate(chars))
indices2chars = dict((i, c) for i, c in enumerate(chars))

# index labels
labels2indices = dict((c, i) for i, c in enumerate(LABELS))
indices2labels = dict((i, c) for i, c in enumerate(LABELS))


# turn sentences into integer sequences
def folds_with_sentences_to_int_sequences(folds):
    fold_index_sequences = []
    for fold in folds:
        sentences = fold
        index_sequences = []
        for sentence in sentences:
            index_sequences_sentence = []
            for char in sentence:
                index_sequences_sentence.append(chars2indices[char])
            index_sequences.append(index_sequences_sentence)
        index_sequences = pad_sequences(index_sequences,
                                        maxlen=max_len, dtype='int32')
        fold_index_sequences.append(index_sequences)
    return fold_index_sequences

X_train = folds_with_sentences_to_int_sequences(input_folds_train_X)
X_test = folds_with_sentences_to_int_sequences(input_folds_test_X)
Y_train = input_folds_train_Y
Y_test = input_folds_test_Y

# Train and test
for fold in range(0, FOLDS_LIMIT):

    # # define and compile model
    model = Sequential()
    model.add(Embedding(num_chars, 256, input_length=max_len))
    model.add(LSTM(output_dim=128, activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['categorical_accuracy'])

    # train model (saving checkpoints)
    checkpointer_all = ModelCheckpoint(
        save_best_only=True, verbose=1,
        filepath=(CHECKPOINTFOLDER +
                  "/weights.{epoch:02d}-{val_loss:.2f}.hdf5"))
    checkpointer_best = ModelCheckpoint(
        save_best_only=True, verbose=1,
        filepath=(CHECKPOINTFOLDER +
                  "/best-weights.hdf5"))
    # early stopping after 15 iterations of no improvement
    early_stopping = EarlyStopping(monitor='val_loss', patience=15,
                                   verbose=1, mode='min')

    model.fit(X_train[fold], Y_train[fold], batch_size=16, nb_epoch=ITERATIONS,
              callbacks=[checkpointer_all, checkpointer_best, early_stopping],
              validation_split=VALIDATION_SET_SIZE, shuffle=True)

    # TEST

    # helper to set only highest predictions to 1, rest to 0
    def get_argmax_predictions(y_pred):
        y_pred_thresholded = np.zeros(y_pred.shape, dtype=np.bool)
        mask = np.argmax(y_pred, axis=1)
        y_pred_thresholded[np.arange(y_pred.shape[0]), mask] = 1
        return y_pred_thresholded

    # # Test with last model
    score = model.evaluate(X_test[fold], Y_test[fold], batch_size=16)
    print()
    logger.info("Test Scores:")
    logger.info("categorical_crossentropy: " + "{:0.3f}".format(score[0]))
    logger.info("categorical_accuracy: " + "{:0.3f}".format(score[1]))
    y_pred = model.predict(X_test[fold])
    mcc_score = mcc(y_pred, Y_test[fold])
    mcc_score_thresholded = mcc(get_argmax_predictions(y_pred), Y_test[fold])
    logger.info("mcc: " + "{:0.3f}".format(mcc_score))
    logger.info("mcc with argmax predictions: " +
                "{:0.3f}".format(mcc_score_thresholded))

    # Test with best model
    model.load_weights(CHECKPOINTFOLDER + "/best-weights.hdf5")
    score = model.evaluate(X_test[fold], Y_test[fold], batch_size=16)
    print()
    logger.info("Test Scores with best model:")
    logger.info("categorical_crossentropy: " + "{:0.3f}".format(score[0]))
    logger.info("categorical_accuracy: " + "{:0.3f}".format(score[1]))
    y_pred = model.predict(X_test[fold])
    mcc_score = mcc(y_pred, Y_test[fold])
    mcc_score_thresholded = mcc(get_argmax_predictions(y_pred), Y_test[fold])
    logger.info("mcc: " + "{:0.3f}".format(mcc_score))
    logger.info("mcc with argmax predictions: " +
                "{:0.3f}".format(mcc_score_thresholded))
