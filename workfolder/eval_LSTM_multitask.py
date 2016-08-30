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
from optparse import OptionParser

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

# --- Parse Command Line Arguments

parser = OptionParser()
parser.add_option("-v", "--verbose", dest="verbose",  action="store_true",
                  default=False, help="Show debug output.")

(options, args) = parser.parse_args()

# Network Architecture
ITERATIONS = 10  # 60
HIDDEN_DIM = 10  # 512
DATA_LIMIT = 100  # None
VALIDATION_SET_SIZE = 0.1
SAMPLING_LENGTH = 50

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
num_labels = len(LABELS)

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
if options.verbose:
    logger.setLevel(logging.DEBUG)
else:
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


def get_confidence_color(confidence):
    # stretch confidence from [0.5, 1] interval to [0,1] interval
    confidence = (confidence - 0.5) * 2
    # get color
    if (np.floor(confidence)) * 6 < 1:
        return color.BLUE0
    if (np.floor(confidence)) * 6 < 2:
        return color.BLUE1
    if (np.floor(confidence)) * 6 < 3:
        return color.BLUE2
    if (np.floor(confidence)) * 6 < 4:
        return color.BLUE3
    if (np.floor(confidence)) * 6 < 5:
        return color.BLUE4
    if (np.floor(confidence)) * 6 < 6:
        return color.BLUE5


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


def prob2conf(probability):
    """ Convert probabilities into confidence statements
    """
    return (probability - (1/LABELS)) * LABELS

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
num_chars = len(chars)
logger.info('Total labels: ' + str(num_labels))
logger.info('Total chars: ' + str(num_chars))

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
charsY2indices = dict((c, i + num_labels) for i, c in enumerate(chars))
indices2charsY = dict((i + num_labels, c) for i, c in enumerate(chars))

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
            if (sentence == data_X[0] and options.verbose):
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

X, Y = vectorize_sequences(sequences_X, next_X, next_Y)

logger.info("Vectorized sequences.")
logger.info("X has dimensions " + str(len(sequences_X)) +
            " x " + str(WIN_LEN) + " x " + str(num_chars) +
            " (#sequences x window length x #chars)")
logger.info("Y has dimensions " + str(len(sequences_X)) +
            " x " + str(num_chars + num_labels) +
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
                   input_shape=(WIN_LEN, num_chars)))
    model.add(Dropout(0.2))
    model.add(LSTM(HIDDEN_DIM, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_chars + num_labels))
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
        1, out_prob_vector[:num_labels], 1))
    label_confidence = out_prob_vector[label_index]
    char_index = np.argmax(np.random.multinomial(
        1, out_prob_vector[num_labels:], 1)) + num_labels
    char_confidence = out_prob_vector[char_index]

    return label_index, label_confidence, char_index, char_confidence


def max_from_output(out_prob_vector):
    '''
    Get char indices from a probability array returned as
    output by the LSTM at prediction time by taking the ones with
    the highest probability
    '''
    label_index = np.argmax(out_prob_vector[:num_labels])
    label_confidence = out_prob_vector[label_index]
    char_index = np.argmax(out_prob_vector[num_labels:]) + num_labels
    char_confidence = out_prob_vector[char_index]

    return label_index, label_confidence, char_index, char_confidence


def LSTM_predict_sentence(model, sentence, debug=False,
                          step_size=WIN_STEP_SIZE):
    '''
    Predict the label of a sentence. Generates sequences from the sentence and
    predicts the label using majority vote over the predictions.
    '''
    sequences, _, _ = sentence2sequences(sentence, step_size=step_size)
    X, _ = vectorize_sequences(sequences)
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


def LSTM_predict_sentences(model, sentences):
    '''
    Predict a batch of sentences and return an indicator matrix
    '''
    # predictions in the format samples x classes
    pred_ind_matrix = np.zeros((len(sentences), num_labels), dtype=np.bool)
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
    mcc_validation = MCC(pred_ind_matrix, Y[:, :num_labels])

    logger.info("MCC: " + "{:0.3f}".format(mcc_validation) +
                ", mean confidence: " + "{:0.3f}".format(np.mean(confidences)))

    if options.verbose:
        sentence = data_X[np.random.randint(0, len(data_X))]
        logger.debug("[1] Labeling random sentence: ")
        label_index, confidence = LSTM_predict_sentence(
            model, str(sentence), debug=True)
        logger.debug("Label: '" + str(indices2labels[label_index]) + "', "
                     "Confidence: " + str(confidence))

        logger.debug("[2] Constructing a sentence: ")

        for diversity in [0.5, 1.0, 2]:

            logger.debug("Seed: Start of sentence above. " +
                         "Diversity: " + str(diversity))

            # placeholder for generated sentence and labels
            gen_sentence = ""
            gen_labels = ""

            # build seed sequence
            sequences, _, _ = sentence2sequences(sentence)
            X, _ = vectorize_sequences(sequences)
            x = np.reshape(X[0], (1, X[0].shape[0], X[0].shape[1]))

            for i in range(SAMPLING_LENGTH):

                preds = model.predict(x, verbose=0)[0]

                (next_label_index_Y_encoded,
                 _,
                 next_char_index_Y_encoded,
                 _) = sample_from_output(preds, diversity)

                gen_sentence += indices2charsY[next_char_index_Y_encoded]
                gen_labels += indices2short_labels[next_label_index_Y_encoded]

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
