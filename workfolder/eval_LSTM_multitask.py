import numpy as np
import re
import random
import sys
import json
import sys
import os
import logging
import pickle
from shutil import copyfile

import numpy as np
import pandas as pd
import scipy

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint

# -----------------------
# SETTINGS
# -----------------------

# Network Architecture
ITERATIONS = 60  # 60
HIDDEN_DIM = 100  # 512
DATA_LIMIT = 100  # None
VALIDATION_SET_SIZE = 0.1
SAMPLING_LENGTH = 400

# Window size and redundancy
WIN_LEN = 40
WIN_STEP_SIZE = 3

PADDING_TOKEN = "_"

LABELS = ["other",
          "candidate",
          "nextsteps",
          "job",
          "benefits",
          "company"]

# use "a" as in applicant since c is already taken
LABELS_SHORT = ["o", "a", "n", "j", "b", "c"]

# -----------------------
# SETUP
# -----------------------

EXP_NAME = __file__.rstrip(".py")

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

CHECKPOINTFOLDER = RESULTFOLDER + "/" + "model-checkpoints"
if not os.path.exists(CHECKPOINTFOLDER):
    os.makedirs(CHECKPOINTFOLDER)

LAST_WEIGHTS_FILE = CHECKPOINTFOLDER + "/model_weights_last.hdf5"

# -- Logging

# create logger
logger = logging.getLogger(EXP_NAME)
logger.setLevel(logging.DEBUG)

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


class color:
    ''' Constants for printing in color and bold in logs / terminal '''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# -----------------------
# HELPERS FUNCTIONS
# -----------------------

# -- Evaluation

def cov(x, y, return_mean_scalar=True):
    """ Covariance Function for Matthews Correlation coefficient below.
        Can return a vector or scalar which is the mean over all classes.
    """

    N = np.shape(x)[0]
    K = np.shape(x)[1]
    x_centered = (x - np.mean(x, 0))
    y_centered = (y - np.mean(y, 0))
    cov = []
    for n in range(0, N):
        cov.append(x_centered[n].T.dot(y_centered[n]))
    # if the mean is the values is to be returned as a scalar:
    if return_mean_scalar:
        cov = np.sum(cov) / K
    return cov


def MCC(x, y, return_mean_scalar=True):
    """ Matthews Correlation coefficient for K classes
        Can return a vector or scalar which is the mean over all classes.
    """
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

    s = return_mean_scalar

    return (cov(x, y, s) / np.sqrt(cov(x, x, s) * cov(y, y, s)))


def MCC_vector(x, y):
    """ Matthews Correlation coefficient for K classes
        Same as above but always returns a vector.
    """
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

    return (cov(x, y, False) / np.sqrt(cov(x, x, False) * cov(y, y, False)))


# -----------------------
# DATASET PREPARATION
# -----------------------

# read and assign data
df_conf = pickle.load(open("data/sentences-dataframe-" +
                           "confidence_greater-0.6.pickle", "rb"))
data_X = np.array(df_conf['0-sentence'])
data_Y = np.array(df_conf['0_label'])

# if DATA_LIMIT is given trim input amount
if DATA_LIMIT is not None:
    data_X = data_X[:(min(len(data_X), DATA_LIMIT))]
    data_Y = data_Y[:(min(len(data_Y), DATA_LIMIT))]

# find all used characters in dataset
all_text = PADDING_TOKEN
for sentence in data_X:
    all_text += sentence

# Assign indices to labels and characters (labels first so they have the
# same indices as in 'labels2indices' e.g. above)
chars = set(all_text)
logger.info('Total labels: ' + str(len(LABELS)))
logger.info('Total chars: ' + str(len(chars)))

# --- Indexing labels and characters (2 way encoding)

# index chars for X
charsX2indices = dict((c, i) for i, c in enumerate(chars))
indices2charsX = dict((i, c) for i, c in enumerate(chars))

# index labels
labels2indices = dict((c, i) for i, c in enumerate(LABELS))
indices2labels = dict((i, c) for i, c in enumerate(LABELS))

# index short labels
short_labels2indices = dict((c, i) for i, c in enumerate(LABELS_SHORT))
indices2short_labels = dict((i, c) for i, c in enumerate(LABELS_SHORT))

# index chars for Y (start indexing after the highest label index
# so labels and chars can be encoded in the same target vector
charsY2indices = dict((c, i + len(LABELS)) for i, c in enumerate(chars))
indices2charsY = dict((i + len(LABELS), c) for i, c in enumerate(chars))

logger.debug("X char endoding: " + str(indices2charsX))
logger.debug("Y label endoding: " + str(indices2labels))
logger.debug("Y char endoding: " + str(indices2charsY))

# --- Building text sequence data

# produce padded sequences and next char and label for each sentence


def sentence2sequences(sentence, label=None, step_size=WIN_STEP_SIZE):
    '''
    Function to generate sequences from a sentence,
    including the next char and label for prediction for each sequence
    '''
    # move window over sentence, prepend padding in beginning
    sequences = []
    next_chars = []
    next_labels = []
    for i in range(-WIN_LEN - 1 + step_size,
                   len(sentence) - WIN_LEN - 1,
                   step_size):
        sequence_x = ""
        # prepend padding
        if (i < 0):
            sequence_x += (PADDING_TOKEN * -i)
        # extract sequence form sentence
        sequence_x += sentence[max(0, i):i + WIN_LEN]
        # add sequence to dataset
        sequences.append(sequence_x)
        # if label is given
        # add next character and it's corresponding label to the data
        if (label is not None):
            next_chars.append(sentence[i + WIN_LEN])
            next_labels.append(indices2short_labels[labels2indices[label]])
            # log first set only for debugging
            if (sentence == data_X[0]):
                logger.debug(sequence_x + color.BOLD +
                             next_chars[len(next_chars) - 1:][0] + " " +
                             next_labels[len(next_labels) - 1:][0] + " [" +
                             label + "]" + color.END)
    return sequences, next_chars, next_labels

logger.info("Generating text sequences...")

sequences_X = []
next_X = []
next_Y = []
for sentence, label in zip(data_X, data_Y):
    sequences, next_chars, next_labels = sentence2sequences(sentence, label)
    sequences_X.extend(sequences)
    next_X.extend(next_chars)
    next_Y.extend(next_labels)

logger.info("Generated " + str(len(sequences_X)) + " text sequences.")

logger.info("Vectorizing sequences into X and Y data...")

def vectorize_sequences(sequences, next_chars=None, next_labels=None):
    # X has dimensionality (num of sequences) x WIN_LEN x (num of chars)
    X = np.zeros((len(sequences), WIN_LEN, len(chars)), dtype=np.bool)
    # Y is used to predict labels and characters simultaneously
    # so it has dimensionality (num of sequences) x (num of chars + num of labels)
    if next_chars is not None and next_labels is not None:
        Y = np.zeros((len(sequences), len(chars) + len(LABELS)), dtype=np.bool)
    else:
        Y = None

    for s, sequence in enumerate(sequences):
        for c, char in enumerate(sequence):
            try:
                X[s, c, charsX2indices[char]] = 1
            except:
                print(sequence)
            if next_chars is not None and next_labels is not None:
                Y[s, charsY2indices[next_chars[s]]] = 1
                Y[s, short_labels2indices[next_labels[s]]] = 1

    return X, Y

X, Y = vectorize_sequences(sequences_X, next_X, next_Y)

logger.info("Vectorized sequences.")
logger.info("X has dimensions " + str(len(sequences_X)) +
            " x " + str(WIN_LEN) + " x " + str(len(chars)) +
            " (#sequences x window length x #chars)")
logger.info("Y has dimensions " + str(len(sequences_X)) +
            " x " + str(len(chars) + len(LABELS)) +
            " (#sequences x #chars + #labels)]")

logger.info("Generating training and validation sets ...")
# generate training and validation set
val_size = int(np.floor(len(X) * VALIDATION_SET_SIZE))
X_train = X[:len(X) - val_size]
Y_train = Y[:len(Y) - val_size]
X_validation = X[len(X) - val_size:]
Y_validation = Y[len(Y) - val_size:]
logger.info("Training set size: " + str(len(X_train)) + " (" +
            str(100 - VALIDATION_SET_SIZE * 100) + "%)")
logger.info("Validation set size: " + str(len(X_validation)) + " (" +
            str(VALIDATION_SET_SIZE * 100) + "%)")


# -----------------------
# MODEL
# -----------------------

logger.info("Initializing LSTM model")

# check if previous model exist and load it, otherwise initialize new model
try:
    logger.info("Previous model found and loaded.")
    model = load_model(RESULTFOLDER + "/trained_model.hdf5")
except OSError:
    logger.info("No previous weights found, random initialization.")
    # build the model: 2 stacked LSTM RNNs
    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, return_sequences=True,
                   input_shape=(WIN_LEN, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(HIDDEN_DIM, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars) + len(LABELS)))
    model.add(Activation('softmax'))
    # save model
    logger.info("Compiling model.")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # model.compile(loss=MCC_vector, optimizer='rmsprop')
    logger.info("Compiled. Saving model as " +
                RESULTFOLDER + "/trained_model.hdf5")
    model.save(RESULTFOLDER + "/trained_model.hdf5")


def sample_from_output(out_prob_vector, temperature=1.0):
    '''
    Sample label and char indices from a probability array returned as
    output by the LSTM at prediction time
    '''
    out_prob_vector = np.log(out_prob_vector) / temperature
    out_prob_vector = np.exp(out_prob_vector) / np.sum(np.exp(out_prob_vector))
    label_index = np.argmax(np.random.multinomial(
        1, out_prob_vector[:len(LABELS)], 1))
    char_index = np.argmax(np.random.multinomial(
        1, out_prob_vector[len(LABELS):], 1)) + len(LABELS)
    return label_index, char_index


def max_from_output(out_prob_vector):
    '''
    Get char indices from a probability array returned as
    output by the LSTM at prediction time by taking the ones with
    the highest probability
    '''
    label_index = np.argmax(out_prob_vector[:len(LABELS)])
    char_index = np.argmax(out_prob_vector[len(LABELS):]) + len(LABELS)
    return label_index, char_index


def LSTM_predict_sentence(model, sentence):
    '''
    Predict the label of a sentence. Generates sequences from the sentence and
    predicts the label using majority vote over the predictions.
    '''
    sequences, _, _ = sentence2sequences(sentence)
    X, _ = vectorize_sequences(sequences)
    predictions = []
    for x in X:
        x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        # predict probabilities of labels and characters with the LSTM
        out_prob_vector = model.predict(x, verbose=0)[0]
        # get max values from probabilities for label and char
        label_index, _ = max_from_output(out_prob_vector)
        predictions.append(label_index)
    # if no predictions where made take random label
    if not predictions:
        label_index = np.random.randint(0, len(LABELS))
        confidence = 0
    else:
        # take mode (= majority vote for most occuring label)
        mode = scipy.stats.mode(predictions)
        label_index = mode.mode[0]  # majority label index
        confidence = mode.count[0] / len(predictions)  # proportion for label
    return label_index, confidence


def LSTM_predict_sentences(model, sentences):
    '''
    Predict a batch of sentences and return an indicator matrix
    '''
    # predictions in the format samples x classes
    pred_ind_matrix = np.zeros((len(sentences), len(LABELS)), dtype=np.bool)
    confidences = []
    for s, sentence in enumerate(sentences):
        label_index, confidence = LSTM_predict_sentence(model, str(sentence))
        pred_ind_matrix[s, label_index] = 1
        confidences.append(confidence)
    return pred_ind_matrix, confidences


# TODO: Train in folds (same folds as other exp) and use training data
#       that's seperate from validation data

# -----------------------
# TRAINING
# -----------------------

# train the model, output generated text after each iteration
for iteration in range(1, ITERATIONS):
    logger.info('Iteration ' + str(iteration) + "/" + str(ITERATIONS))

# TODO: record which fold for checkpoints when implementing CV
    checkpointer = ModelCheckpoint(
        filepath=(CHECKPOINTFOLDER + "/model-weights" +
                  ".{epoch:02d}-{val_loss:.2f}.hdf5"), verbose=1)
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=1,
              validation_data=(X_validation, Y_validation),
              callbacks=[checkpointer])
    # save status
    model.save(RESULTFOLDER + "/trained_model.hdf5")

    # score validation set sentences
    pred_ind_matrix, confidences = LSTM_predict_sentences(
        model, data_X[len(data_X) - val_size:])
    mcc_validation = MCC(pred_ind_matrix, Y[:, :len(LABELS)])

    logger.info("MCC: " + "{:0.3f}".format(mcc_validation) +
                ", mean confidence: " + "{:0.3f}".format(np.mean(confidences)))

    print(pred_ind_matrix.shape)
    print(Y[:, :len(LABELS)].shape)

    # for debugging sample predictions from the model

    # start_index = random.randint(0, len(all_text) - maxlen - 1)
    #
    # print('----- classifying text')
    # print()
    #
    # for diversity in [1.0]:
    #
    #     print()
    #     print('----- diversity:', diversity)
    #
    #     rnn_current_chars  = all_text[start_index: start_index + maxlen]
    #     rnn_current_labels = all_labels[start_index: start_index + maxlen]
    #
    #     chars_true  = all_text[start_index: start_index + maxlen
    #                                          + SAMPLING_LENGTH]
    #     labels_true = all_labels[start_index: start_index + maxlen
    #                                           + SAMPLING_LENGTH]
    #
    #     generated_labels = ''
    #     generated_labels += rnn_current_labels
    #
    #     print('----- Generating with seed: "' + rnn_current_chars + '"')
    #
    #     for i in range(SAMPLING_LENGTH):
    #         x = np.zeros((1, maxlen, len(chars)))
    #         for t, char in enumerate(rnn_current_chars):
    #             x[0, t, char_indices[char]] = 1.
    #
    #         preds = model.predict(x, verbose=0)[0]
    #
    #         next_label = sample_label(preds, diversity)
    #
    #         generated_labels += str(next_label)
    #
    #         rnn_current_chars  = chars_true[maxlen + i:i + 2*maxlen]
    #         rnn_current_labels = rnn_current_labels[1:] + str(next_label)
    #
    #     # print generated chars and labels with true labels
    #     print_pred(generated_labels, chars_true, labels_true, 70)
    #
    #     # calculate and print accuracy
    #     accuracy = label_accuracy(generated_labels, labels_true)
    #     print("Label Accuracy on sampled text:", accuracy)
    #
    #     # calculate MCC on validation data:
    #     y_all_true = []
    #     y_all_pred = []
    #     for i, y_true in enumerate(y_test):
    #         y_pred = model.predict(X_test[i:i+1], verbose=0)[0]
    #         y_all_pred.append(y_pred)
    #         y_all_true.append(y_true)
    #     print("MCC:", MCC(np.vstack(y_all_true), np.vstack(y_all_pred)))
