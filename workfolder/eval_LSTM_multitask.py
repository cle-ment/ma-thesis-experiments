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
from shutil import rmtree
from optparse import OptionParser
from copy import deepcopy

from subprocess import call

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras import backend as K

# ------------------------------------------------------------------------------
# SETTINGS AND SETUP
# ------------------------------------------------------------------------------

# --- Parse Command Line Arguments

parser = OptionParser()
parser.add_option("-v", "--verbose", dest="verbose",  action="store_true",
                  default=False, help="Show debug output.")
parser.add_option("-s", "--sync", dest="sync",  action="store_true",
                  default=False, help="Sync results to server.")
parser.add_option("-t", "--testrun", dest="testrun",  action="store_true",
                  default=False, help="Only do a test run with some data.")

(options, args) = parser.parse_args()

# Network Architecture
ITERATIONS = 60  # 60
HIDDEN_DIM = 512  # 512
DATA_LIMIT_TRAIN = None  # None, limit the number of used test data
DATA_LIMIT_TEST = None  # None, limit the number of used test data
FOLD_LIMIT = 1  # None, limit the number of processed folds
VALIDATION_SET_SIZE = 0.05
SAMPLING_LENGTH = 60

# Window size and redundancy
WIN_LEN = 40
WIN_STEP_SIZE = 1

# If test run overwrite parameters
if options.testrun:
    # testing parameters
    ITERATIONS = 5  # 60
    HIDDEN_DIM = 10  # 512
    DATA_LIMIT_TRAIN = 100  # None, limit the number of used test data
    DATA_LIMIT_TEST = 50  # None, limit the number of used test data
    FOLD_LIMIT = 1  # None, limit the number of processed folds
    VALIDATION_SET_SIZE = 0.1
    SAMPLING_LENGTH = 30
    WIN_LEN = 40
    WIN_STEP_SIZE = 3


PADDING_TOKEN = "_"

LABELS = ["other",
          "candidate",
          "nextsteps",
          "job",
          "benefits",
          "company"]
num_labels = len(LABELS)

# use "a" as in applicant since c is already taken
LABELS_SHORT = ["o", "a", "n", "j", "b", "c"]


# -- colors for terminal output

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
    BLUE0 = '\033[48;5;16m'
    BLUE1 = '\033[48;5;17m'
    BLUE2 = '\033[48;5;18m'
    BLUE3 = '\033[48;5;19m'
    BLUE4 = '\033[48;5;20m'
    BLUE5 = '\033[48;5;21m'


# --- SETUP

# -- Basefolder setup

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

# -- Logging

# create logger
logger = logging.getLogger(EXP_NAME)


# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
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

# create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)s | %(message)s')

# add formatter to ch
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

logger.propagate = False


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------


# --- UTILITY FUNCTIONS

# -- Evaluation

def matthews_correlation(y_true, y_pred):
    ''' Matthews correlation coefficient
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(1 - y_neg * y_pred_pos)
    fn = K.sum(1 - y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def cov(x, y, return_mean_scalar=True):
    """ Covariance Function for Matthews Correlation coefficient below.

    Args:
        x: Real valued matrix (samples x classes) with class assigments
        y: Real valued matrix (samples x classes) with class assigments
        return_mean_scalar: if True will function return a scalar,
            otherwise a vector

    Returns:
        The covariance value or vector
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
    """ Matthews Correlation coefficient for K classes.

    Args:
        x: Real valued matrix (samples x classes) with class assigments
        y: Real valued matrix (samples x classes) with class assigments
        return_mean_scalar: if True will function return a scalar,
            otherwise a vector

    Returns:
        Matthews Correlation Coefficient as a real value or real-valued vector
    """
    # check if the input is in multiclass form
    try:
        y_type, _, _ = sklearn.metrics.classification._check_targets(x, y)
        print(y_type)
        if y_type.startswith('multiclass'):
            x = lb.transform(x)
            y = lb.transform(y)
    except:
        msg = ('Warning, input not sklearn compatible: ' +
               'No metrics support "multiclass-multioutput" format')
        logger.warning(msg)

    s = return_mean_scalar
    mcc = (cov(x, y, s) / np.sqrt(cov(x, x, s) * cov(y, y, s)))

    return mcc


def MCC_vector(x, y):
    """ Matthews Correlation coefficient for K classes. Same as above
        but always returns a vector.

    Args:
        x: Real valued matrix (samples x classes) with class assigments
        y: Real valued matrix (samples x classes) with class assigments

    Returns:
        Matthews Correlation Coefficients as vector
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

    mcc = (cov(x, y, False) / np.sqrt(cov(x, x, False) * cov(y, y, False)))

    return mcc


# -- misc utility functions

def get_confidence_color(confidence, num_classes=num_labels):
    """ Generate a color given confidence of the classifier

    Args:
        confidence: Real-valued confidence in interval [0,1]

    Returns:
        Color given the confidence. Brighter blues for higher confidence.
    """
    # get color
    if (np.floor(confidence)) * num_classes < 2:
        return color.BLUE1
    if (np.floor(confidence)) * num_classes < 3:
        return color.BLUE2
    if (np.floor(confidence)) * num_classes < 4:
        return color.BLUE3
    if (np.floor(confidence)) * num_classes < 5:
        return color.BLUE4
    if (np.floor(confidence)) * num_classes < 6:
        return color.BLUE5
    else:
        return color.BLUE0


def prob2conf(probability, num_classes=num_labels):
    """ Convert probabilities into confidence statements.
        Depends on the number of classes.
        of classes / labels.

    Args:
        probability: A probability value produced by the classifier.

    Returns:
        A confidence value in the interval [0, 1] from the given probability.
    """
    return (probability - (1/num_labels)) * num_labels


def store_results(
    computation_progress,
    src=("/home/ubuntu/thesis-experiments/output/" + EXP_NAME + "/"),
        trg="clemens@cwestrup.de:thesis/output/" + EXP_NAME):
    """ Store and sync the results. Should be executed at every checkpoint.

    Args:
        computation_progress: Dictionary with indicators of the computation
            progress stored in it
        src: Source directory for syncing the results
        trg: Target directory for syncing the results
    """
    pickle.dump(computation_progress,
                open(BASE_OUTFOLDER + "/computation_progress.pickle", "wb"))
    if options.sync:
        call(["rsync", "-av", "--update", "--delete", "--force", src, trg])


# --- DATASET PROCESSING

def sentence2sequences(sentence, label=None,
                       step_size=WIN_STEP_SIZE, debug=False):
    """ Function to generate sequences from a sentence,
        including the next char and label for prediction for each sequence.

    Args:
        sentence: A sentence to be converted into sequences.
        label: The class label for the sentence (only given for training data).
            Should be in the form of LABELS constant defined above.
        step_size: The step_size of the window for producing sequences.
        debug: Print sequences

    Returns:
        sequences: A list of sequences produced from the sentence
        next_chars: If argument 'label' is given a list of the chars following
            the last char of each of the sequences. Otherwise an empty list.
        next_labels: If argument 'label' is given: A list of the labels
            for each of the sequences. Otherwise an empty list.
    """
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
            if debug:
                logger.debug(sequence_x + color.BOLD +
                             next_chars[len(next_chars) - 1:][0] + " " +
                             next_labels[len(next_labels) - 1:][0] + " [" +
                             label + "]" + color.END)
    return sequences, next_chars, next_labels


def vectorize_sequences(sequences, next_chars=None, next_labels=None):
    """ Vectorize a set of text sequences into format the network can process

    Args:
        sequences: Sequences to vectorize. Must be of lenght WIN_LEN
        next_chars: List of the chars following each sequence (only training).
        next_labels: List of the label for each sequence (only training).

    Returns:
        X: Indicator tensor of dimensions (#sequences x WIN_LEN, num_chars)
        Y: Indicator tensor of dimensions (#sequences x num_chars + num_labels)
            if arguments 'next_chars' and 'next_labels' were given. Otherwise
            None.
    """
    # X has dimensions: (num of sequences) x WIN_LEN x (num of chars)
    X = np.zeros((len(sequences), WIN_LEN, num_chars), dtype=np.bool)
    # Y is used to predict labels and characters simultaneously
    # so it has dimensions: (num of sequences) x (num of chars + num of labels)
    if next_chars is not None and next_labels is not None:
        Y = np.zeros((len(sequences), num_chars + num_labels), dtype=np.bool)
    else:
        Y = None

    for s, sequence in enumerate(sequences):
        for c, char in enumerate(sequence):
            try:
                X[s, c, charsX2indices[char]] = 1
            except:
                logger.error("Can't vectorize sequence: " + sequence)
                sys.exit()
            if next_chars is not None and next_labels is not None:
                Y[s, charsY2indices[next_chars[s]]] = 1
                Y[s, short_labels2indices[next_labels[s]]] = 1

    return X, Y


# --- LSTM MODEL FUNCTIONS

def sample_from_output(out_prob_vector, temperature=1.0):
    """ Sample label and char indices from a probability array returned as
        output by the LSTM at prediction time

    Args:
        out_prob_vector: Output distribution from the LSTM
        temperature: 'Flattening' factor of distribution, making unlikely
            outcomes more likely.

    Returns:
        label_index: Index for the sampled label
        label_confidence: Sample confidence for label, based on probability
        char_index: Index for the sampled char
        char_confidence: Sample confidence for char, based on probability
    """
    out_prob_vector = np.log(out_prob_vector) / temperature
    out_prob_vector = np.exp(out_prob_vector) / np.sum(np.exp(out_prob_vector))
    label_index = np.argmax(np.random.multinomial(
        1, out_prob_vector[:num_labels], 1))
    label_confidence = prob2conf(out_prob_vector[label_index])
    char_index = np.argmax(np.random.multinomial(
        1, out_prob_vector[num_labels:], 1)) + num_labels
    char_confidence = out_prob_vector[char_index]
    if char_confidence > 1:
        logger.error("Char confidence over 1: " + char_confidence)

    return label_index, label_confidence, char_index, char_confidence


def LSTM_predict_sentence(model, sentence, debug=False,
                          step_size=WIN_STEP_SIZE):
    """ Predict the label of a sentence. Generates sequences from the
        sentence and predicts the label using majority vote over the
        predictions.

    Args:
        model: LSTM model used for prediction
        sentence: A sample to predict the next character and label for
        debug: Show debug output or not
        step_size: Step size of the window
    """

    # TODO:0 Weigh predictions by their confidences
    # TODO:10 Weigh predictions by their prior amount

    sequences, _, _ = sentence2sequences(sentence, step_size=step_size)
    X, _ = vectorize_sequences(sequences)

    # print(X)

    # x = np.reshape(X, (1, X.shape[0], X.shape[1]))

    return LSTM_predict_sentence_sequence_vectors(
        model, X, debug=debug, step_size=step_size)


# TODO: Remove?
# def LSTM_predict_sentences(model, sentences):
#     """ Predict a set of sentences and return an indicator matrix
#
#     Args:
#         model: LSTM model to use for prediction.
#         sentences: Sentences to predict upon.
#
#     Returns:
#         pred_ind_matrix: Indicator matrix of predictions (samples x classes)
#         confidences: List of confidences for predictions
#     """
#     # predictions in the format samples x classes
#     pred_ind_matrix = np.zeros((len(sentences), num_labels), dtype=np.bool)
#     confidences = []
#     for s, sentence in enumerate(sentences):
#         label_index, confidence = LSTM_predict_sentence(model, str(sentence))
#         pred_ind_matrix[s, label_index] = 1
#         confidences.append(confidence)
#     return pred_ind_matrix, confidences


def LSTM_predict_sentence_sequence_vectors(model, X, debug=False,
                          step_size=WIN_STEP_SIZE):

    # TODO: comment this function

    predictions = []
    confidences = []

    for x in X:
        x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        # predict probabilities of labels and characters with the LSTM
        out_prob_vector = model.predict(x, verbose=0)[0]

        # get max values from probabilities for label and char
        (label_index,
         label_confidence, _, _) = sample_from_output(out_prob_vector)
        predictions.append(label_index)
        confidences.append(label_confidence)
    if debug:
        # TODO: don't print a whole black newline for long sentences
        filler = " " * int(step_size - 1)
        confidence_colored_labels = ""
        for p, prediction in enumerate(predictions):
            confidence_color = get_confidence_color(confidences[p])
            confidence_colored_labels += color.BLUE0 + filler + color.END
            confidence_colored_labels += confidence_color
            confidence_colored_labels += str(indices2short_labels[prediction])
            confidence_colored_labels += color.END
        logger.debug(sentence)
        logger.debug(confidence_colored_labels)
    # if no predictions where made take random label
    if not predictions:
        label_index = np.random.randint(0, num_labels)
        confidence = 0
    else:
        # take mode (= majority vote for most occuring label)
        mode = scipy.stats.mode(predictions)
        label_index = mode.mode[0]  # majority label index
        confidence = mode.count[0] / len(predictions)  # proportion for label
    return label_index, confidence


# ------------------------------------------------------------------------------
# PREPERATION, TRAINING AND TESTING OF MODEL
# ------------------------------------------------------------------------------

# -- Init with current computation progress

try:
    computation_progress = pickle.load(
        open(BASE_OUTFOLDER + "/computation_progress.pickle", "rb"))
except:
    # no computation was done yet so create a progress file_prefix
    computation_progress = {
        'fold': 0,
        'iteration': 0,
        'results': [],
        'current_folds_train': [],
        'current_folds_test': []
    }
    store_results(computation_progress)


# --- DATASET PREPARATION

# -- Load data

input_folds_train_X = pickle.load(
    open(INPUTFOLDER + "/folds_train_X.pickle", "rb"))
input_folds_train_Y = pickle.load(
    open(INPUTFOLDER + "/folds_train_Y.pickle", "rb"))
input_folds_test_X = pickle.load(open(
    INPUTFOLDER + "/folds_test_X.pickle", "rb"))
input_folds_test_Y = pickle.load(open(
    INPUTFOLDER + "/folds_test_Y.pickle", "rb"))

# Load label binarizer
lb = pickle.load(open(INPUTFOLDER + "/lb.pickle", "rb"))

# -- Trim data (if limits are given)

# if FOLD_LIMIT is given limit the number of folds used
given_num_folds = len(input_folds_train_X)
num_folds = given_num_folds
if FOLD_LIMIT is not None:
    num_folds = min(FOLD_LIMIT, given_num_folds)
    input_folds_train_X = input_folds_train_X[:num_folds]
    input_folds_test_X = input_folds_test_X[:num_folds]
    input_folds_train_Y = input_folds_train_Y[:num_folds]
    input_folds_test_Y = input_folds_test_Y[:num_folds]
    logger.debug("Using " + str(num_folds) + " of " + str(given_num_folds) +
                 " total folds (limit imposed by setting FOLD_LIMIT)")

# if DATA_LIMIT is given only use one fold and trim input amount
if DATA_LIMIT_TRAIN is not None:
    logger.debug("Training data limited to " + str(DATA_LIMIT_TRAIN) +
                 " (limit imposed by setting DATA_LIMIT_TRAIN)")
    for fold in range(0, num_folds):
        input_folds_train_X[fold] = input_folds_train_X[fold][
            :(min(len(input_folds_train_X[fold]), DATA_LIMIT_TRAIN))]
        input_folds_train_Y[fold] = input_folds_train_Y[fold][
            :(min(len(input_folds_train_Y[fold]), DATA_LIMIT_TRAIN))]
if DATA_LIMIT_TEST is not None:
    logger.debug("Test data limited to " + str(DATA_LIMIT_TEST) +
                 " (limit imposed by setting DATA_LIMIT_TEST)")
    for fold in range(0, num_folds):
        input_folds_test_X[fold] = input_folds_test_X[fold][
            :(min(len(input_folds_test_X[fold]), DATA_LIMIT_TEST))]
        input_folds_test_Y[fold] = input_folds_test_Y[fold][
            :(min(len(input_folds_test_Y[fold]), DATA_LIMIT_TEST))]

# -- process chars

# find all used characters in dataset
all_folds_X = deepcopy(input_folds_train_X)
all_folds_X.extend(input_folds_test_X)
all_text = PADDING_TOKEN
for fold_X in all_folds_X:
    for sentence in fold_X:
        all_text += sentence
del all_folds_X

# count chars and labels
chars = set(all_text)
num_chars = len(chars)
logger.info('Total labels: ' + str(num_labels))
logger.info('Total chars: ' + str(num_chars))


# -- Index labels and characters (2 way encoding)

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
charsY2indices = dict((c, i + num_labels) for i, c in enumerate(chars))
indices2charsY = dict((i + num_labels, c) for i, c in enumerate(chars))

logger.debug("X char endoding: " + str(indices2charsX))
logger.debug("Y label endoding: " + str(indices2labels))
logger.debug("Y char endoding: " + str(indices2charsY))


# -- Produce sequences from text

# produce padded sequences and next char and label for each sentence
logger.info("Generating text sequences...")

# print first sentence for debugging
if options.verbose:
    logger.debug("Sequencing example (first sentence):")
    logger.debug(input_folds_train_X[0][0])


def folds2sequences(input_folds_X, input_folds_Y,
                    debug=False, test_data=False):

# TODO: Comment function
# test_data: group sequencens by sentence

    is_first_sentence = debug
    num_seq = 0
    folds_sequences = []
    folds_next_chars = []
    folds_next_labels = []

    for fold_X, fold_Y in zip(input_folds_X, input_folds_Y):

        fold_sequences = []
        fold_next_chars = []
        fold_next_labels = []

        # reverse transform Y to get label names back
        fold_Y = lb.inverse_transform(fold_Y)

        # create sequences and Y data (next_chars and next_labels) foreach fold
        for sentence, label in zip(fold_X, fold_Y):
            sequences, next_chars, next_labels = sentence2sequences(
                sentence, label, debug=is_first_sentence)

            if test_data:
                fold_sequences.append(sequences)
                fold_next_chars.append(next_chars)
                fold_next_labels.append(next_labels)
            else:
                fold_sequences.extend(sequences)
                fold_next_chars.extend(next_chars)
                fold_next_labels.extend(next_labels)

            num_seq += len(sequences)
            is_first_sentence = False

        folds_sequences.append(fold_sequences)
        folds_next_chars.append(fold_next_chars)
        folds_next_labels.append(fold_next_labels)

    return folds_sequences, folds_next_chars, folds_next_labels, num_seq

num_sequences = 0

# training folds sequences
(folds_sequences_train,
 folds_next_chars_train,
 folds_next_labels_train,
 num_seq_train) = folds2sequences(input_folds_train_X,
                                  input_folds_train_Y,
                                  debug=options.verbose)
num_sequences += num_seq_train

# test folds sequences
(folds_sequences_test,
 folds_next_chars_test,
 folds_next_labels_test,
 num_seq_test) = folds2sequences(input_folds_test_X,
                                 input_folds_test_Y,
                                 test_data=True)
num_sequences += num_seq_test

logger.info("Generated " + str(num_sequences) + " text sequences in total.")

# -- Vectorize sequences

logger.info("Vectorizing sequences...")

data_folds_train_X = []
data_folds_test_X = []
data_folds_train_Y = []

# training data
for fold in range(0, num_folds):
    X, Y = vectorize_sequences(folds_sequences_train[fold],
                               folds_next_chars_train[fold],
                               folds_next_labels_train[fold])
    data_folds_train_X.append(X)
    data_folds_train_Y.append(Y)

# test data
num_test_sequences_folds = []
for fold in range(0, num_folds):
    num_test_sequences = 0
    Xs = []
    for sentence_i in range(0, len(folds_sequences_test[fold])):
        X, _ = vectorize_sequences(folds_sequences_test[fold][sentence_i],
                                   folds_next_chars_test[fold][sentence_i],
                                   folds_next_labels_test[fold][sentence_i])
        Xs.append(X)
        num_test_sequences += len(X)
    data_folds_test_X.append(Xs)
    num_test_sequences_folds.append(num_test_sequences)


logger.info("Vectorized sequences.")
logger.debug("X dimensions: #sequences x window length x #chars")
logger.debug("Fold 1, train: " + str(len(data_folds_train_X[0])) +
             " x " + str(WIN_LEN) + " x " + str(num_chars))
logger.debug("Fold 1, test:  " + str(num_test_sequences_folds[0]) +
             " x " + str(WIN_LEN) + " x " + str(num_chars))
logger.debug("Y dimensions train: #sequences x #chars + #labels")
logger.debug("Fold 1, train: " + str(len(data_folds_train_Y[0])) +
             " x " + str(num_chars + num_labels))
logger.debug("Y dimensions test: #sentences x #labels")
logger.debug("Fold 1, test:  " + str(len(input_folds_test_Y[0])) +
             " x " + str(num_labels))


# --- TRAINING, VALIDATION AND TESTING

# store test scores for each fold
mcc_test_scores = []

# train and test a model on each fold
for fold in range(0, num_folds):

    # skip fold if it's already computed
    if computation_progress['fold'] > fold:
        continue

    logger.info('### Fold ' + str(fold + 1) + "/" + str(num_folds))

    # -- Model Setup

    logger.info("Initializing LSTM model")

    # check if previous model exist and load it, otherwise initialize new model
    trained_model_file = (RESULTFOLDER + "/trained_model_fold" +
                          str(fold) + ".hdf5")

    try:
        logger.info("Previous model found and loaded.")
        model = load_model(trained_model_file)
    except OSError:
        logger.info("No previous weights found, random initialization.")
        # build the model: 2 stacked LSTM RNNs
        model = Sequential()
        model.add(LSTM(HIDDEN_DIM, return_sequences=True,
                       input_shape=(WIN_LEN, num_chars)))
        model.add(Dropout(0.2))
        model.add(LSTM(HIDDEN_DIM, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(num_chars + num_labels))
        model.add(Activation('softmax'))
        # save model
        logger.info("Compiling model. Saving as " + trained_model_file)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=[matthews_correlation])
        # model.compile(loss=MCC_vector, optimizer='rmsprop')
        model.save(trained_model_file)

    # -- TRAINING / VALIDATION
    # train the model, output generated text after each iteration
    for iteration in range(0, ITERATIONS):

        # skip iteration if it's already computed
        if computation_progress['iteration'] > iteration:
            continue

        logger.info('## Iteration ' + str(iteration + 1) +
                    "/" + str(ITERATIONS))

        checkpointer = ModelCheckpoint(
            filepath=(CHECKPOINTFOLDER + "/model-weights" +
                      ".{epoch:02d}-{val_loss:.2f}.hdf5"), verbose=1)
        model.fit(data_folds_train_X[fold], data_folds_train_Y[fold],
                  batch_size=128, nb_epoch=1,
                  validation_split=VALIDATION_SET_SIZE, shuffle=True,
                  callbacks=[checkpointer])
        # save status
        model.save(RESULTFOLDER + "/trained_model" + str(fold) + ".hdf5")

        # -- Output for debugging and monitoring:
        # 1. Label a random sentence from the data
        # 2. Produce a labelled sentence from scratch, given a seed
        if options.verbose:
            sentence = input_folds_train_X[fold][
                np.random.randint(0, len(input_folds_train_X[fold]))]
            logger.debug("[1] Labeling random sentence: ")
            label_index, confidence = LSTM_predict_sentence(
                model, str(sentence), debug=True)
            logger.debug("Label: '" + str(indices2labels[label_index]) + "', "
                         "Confidence: " + str(confidence))

            logger.debug("[2] Constructing a sentence: ")

            # build seed sequence
            sequences, _, _ = sentence2sequences(sentence)
            X, _ = vectorize_sequences(sequences)

            logger.debug("Seed: " + sequences[0])

            for diversity in [0.5, 1.0, 2]:

                logger.debug("Diversity: " + str(diversity))

                # reset x to original sequence
                x = np.reshape(X[0], (1, X[0].shape[0], X[0].shape[1]))

                # placeholder for generated sentence and labels
                gen_sentence = ""
                gen_labels = ""

                for i in range(SAMPLING_LENGTH):

                    # predict distribution for next char and label
                    preds = model.predict(x, verbose=0)[0]

                    # sample next char and label from distribution
                    (next_label_index_Y_encoded,
                     _,
                     next_char_index_Y_encoded,
                     _) = sample_from_output(preds, diversity)

                    # append next char and label to output
                    gen_sentence += indices2charsY[next_char_index_Y_encoded]
                    gen_labels += indices2short_labels[
                        next_label_index_Y_encoded]

                    # move input sequence forward, FILO style:
                    # kick out first character
                    x = x[:, 1:, :]
                    # attach a new space for a character
                    x = np.hstack([x, np.zeros((1, 1, num_chars))])
                    # add the newly generated character
                    next_char_index_X_encoded = (
                        int(next_char_index_Y_encoded) - num_labels)
                    x[0, len(x), next_char_index_X_encoded] = 1

                logger.debug(gen_sentence)
                logger.debug(color.BLUE0 + gen_labels + color.END)

        # update status
        computation_progress['iteration'] = iteration + 1
        store_results(computation_progress)

    # -- TESTING SCORE

    # TODO: Predict all sequences as one batch and keep index
    # about which sequences (i.e. how many) belong to each sentence. That
    # way the algorithm is called once (in parallel on a big batch) and
    # the score can be calcualted using the results

    # loop over X (each entry are the vectorized sequences for one sentence
    # and Y (the labels for each sentece)
    Y_pred = []
    for X, Y in zip(data_folds_test_X[fold], input_folds_test_Y[fold]):

        # make predictions with the sequences
        label_index, confidence = LSTM_predict_sentence_sequence_vectors(
            model, X)

        Y_pred.append(indices2labels[label_index])

    Y_pred_binarized = lb.transform(Y_pred)

    mcc = MCC(Y_pred_binarized, input_folds_test_Y[fold])
    mcc_test_scores.append(mcc)

    logger.info("MCC: " + "{:0.3f}".format(mcc))

    # update status
    computation_progress['fold'] = fold + 1
    store_results(computation_progress)
