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

import gensim


# --- Basic setup

TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")
VALIDATION_SIZE = 0.2
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


# --- Dataset Initialization

# read data
df = pd.read_csv("./data/sentences_aggregated_50-449.csv")

# Shuffle data
df = df.iloc[np.random.permutation(np.arange(len(df)))]

# Use entries with label confidence over 0.6 and aren't test questions:
df_conf = df[df['0_label:confidence'] > 0.6]
df_conf = df_conf[df_conf['_golden'] == False]
df_conf = df_conf[['0_label', '0_label:confidence', '0-sentence',
                   '0-context-after', '0-context-before']]

# binarize labels
label_array = np.array(df_conf['0_label'])
lb = sklearn.preprocessing.LabelBinarizer()
lb.fit(label_array)
pickle.dump(lb, open("sentences_lb.pickle", "wb"))

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

pickle.dump(folds_train_X, open("sentences_folds_train_X.pickle", "wb"))
pickle.dump(folds_train_Y, open("sentences_folds_train_Y.pickle", "wb"))
pickle.dump(folds_test_X, open("sentences_folds_test_X.pickle", "wb"))
pickle.dump(folds_test_Y, open("sentences_folds_test_Y.pickle", "wb"))


# --- Feature Spaces

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


def score(estimator, X_train, X_train_inferred, X_test_inferred,
          y_train, y_test):

    """ Evaluate the performance on given test and training data using the
        estimator (classifier) provided
    """

    estimator.fit(X_train, y_train)
    predictions_train = estimator.predict(X_train)
    predictions_test = estimator.predict(X_test_inferred)
    mcc_train = MCC(predictions_train, y_train)
    mcc_test = MCC(predictions_test, y_test)

    estimator.fit(X_train_inferred, y_train)
    predictions_train = estimator.predict(X_train_inferred)
    predictions_test = estimator.predict(X_test_inferred)
    mcc_train_inferred = MCC(predictions_train, y_train)
    mcc_test_inferred = MCC(predictions_test, y_test)

    return mcc_train, mcc_test, mcc_train_inferred, mcc_test_inferred


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
    model, corpus_train, corpus_test, y_train, y_test,
    estimator=sklearn.linear_model.LinearRegression(),
    steps=100, alpha_start=0.025, alpha_end=0.0001,
    infer_steps=5, infer_min_alpha=0.0001, infer_alpha=0.1,
        evaluate=True):

    """ Train the given doc2vec model and and evaluate it at each step. This
        measures the performance whent trained on the learned paragraph
        vectors (train_scores) and the test performance when then inferring
        vectors (test_scores) as well as when trained on inferred vectors on
        the training data (train_infer_scores) with the subsequent performance
        on the test data (test_infer_scores)
    """

    # measure start time
    start_time = time.time()

    # set learning rate
    alpha = alpha_start
    alpha_delta = (alpha - alpha_end) / steps

    # store scores during training
    mcc_train = 0
    mcc_test = 0
    train_scores = []
    test_scores = []
    train_infer_scores = []
    test_infer_scores = []

    # store and return the best model
    best_model = None
    best_model_infer = None

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

        # evaluate
        if evaluate:
            # training vectors
            features_train = np.matrix(model.docvecs)
            # inferred training vectors
            features_train_inferred = infer_vectors(
                model, corpus_train_fixed,
                infer_steps, infer_min_alpha, infer_alpha)
            # inferred test vectors
            features_test_inferred = infer_vectors(
                model, corpus_test, infer_steps, infer_min_alpha, infer_alpha)
            # calculate scores
            mcc_train, mcc_test, mcc_train_inferred, mcc_test_inferred = score(
                estimator, features_train, features_train_inferred,
                features_test_inferred, y_train, y_test)
            train_scores.append(mcc_train)
            test_scores.append(mcc_test)
            train_infer_scores.append(mcc_train_inferred)
            test_infer_scores.append(mcc_test_inferred)

            if (np.max(test_scores[:step]) < mcc_test):
                best_model = copy.deepcopy(model)
            if (np.max(test_infer_scores[:step]) < mcc_test_inferred):
                best_model_infer = copy.deepcopy(model)

        # elapsed time
        now_time = time.time()
        time_elapsed = now_time-start_time
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        msg = ("Step " + str(step + 1) + "/" + str(steps) +
               " MCC[" + "{:0.3f}".format(mcc_train) +
               "|" + "{:0.3f}".format(mcc_test) + "]" +
               " MCCinf[" + "{:0.3f}".format(mcc_train_inferred) +
               "|" + "{:0.3f}".format(mcc_test_inferred) + "]")
        print(msg, flush=False, end='\r')

    return (train_scores, test_scores,
            train_infer_scores, test_infer_scores,
            best_model, best_model_infer,
            time_elapsed)


# - PV-DBOW

logger.info("Generating PV-DBOW")

# used classifier: logistic regression
estimator_logreg = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression())

features_pvdbow_train = []
features_pvdbow_test = []

for i in range(0, num_folds):
    # use best settings from previous experiments
    model = gensim.models.Doc2Vec(
        dm=0, size=300, window=10, negative=3, min_count=2,
        hs=0, sample=1e-5, workers=CORES)

    (train_scores, test_scores,
     train_infer_scores, test_infer_scores,
     best_model, best_model_infer,
     time_elapsed) = train_eval_doc2vec(
        model, read_corpus(folds_train_X[i]), read_corpus(folds_test_X[i]),
        folds_train_Y[i], folds_test_Y[i],
        estimator=estimator_logreg,
        steps=150, alpha_start=0.025, alpha_end=0.0001,
        infer_steps=5, infer_min_alpha=0.0001, infer_alpha=0.1,
        evaluate=True)

    trained_model = None

    # check whether infer model is better or not
    if np.max(test_scores) > np.max(test_infer_scores):
        trained_model = best_model
        # used trained vectors of the model
        features_pvdbow_train.append(np.matrix(model.docvecs))
        logger.info("Using training vectors")
    else:
        trained_model = best_model_infer
        # infer training vectors on training data
        features_pvdbow_train.append(infer_vectors(
            trained_model, read_corpus(folds_train_X[i]), 10, 0.0001, 0.1))
        logger.info("Using inferred training vectors")

    # infer training vectors on test data
    features_pvdbow_train.append(infer_vectors(
        trained_model, read_corpus(folds_test_X[i]), 10, 0.0001, 0.1))


pickle.dump(features_pvdbow_train,
            open("sentences_features_pvdbow_train.pickle", "wb"))
pickle.dump(features_pvdbow_test,
            open("sentences_features_pvdbow_test.pickle", "wb"))

logger.info("Done")

# - PV-DM

logger.info("Generating PV-DM")

# used classifier: logistic regression
estimator_logreg = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression())

features_pvdm_train = []
features_pvdm_test = []

for i in range(0, num_folds):
    # use best settings from previous experiments
    model = gensim.models.Doc2Vec(
        dm=1, size=300, window=10, negative=3, min_count=2,
        hs=0, sample=1e-5, workers=CORES)

    (train_scores, test_scores,
     train_infer_scores, test_infer_scores,
     best_model, best_model_infer,
     time_elapsed) = train_eval_doc2vec(
        model, read_corpus(folds_train_X[i]), read_corpus(folds_test_X[i]),
        folds_train_Y[i], folds_test_Y[i],
        estimator=estimator_logreg,
        steps=150, alpha_start=0.025, alpha_end=0.0001,
        infer_steps=5, infer_min_alpha=0.0001, infer_alpha=0.1,
        evaluate=True)

    trained_model = None

    # check whether infer model is better or not
    if np.max(test_scores) > np.max(test_infer_scores):
        trained_model = best_model
        # used trained vectors of the model
        features_pvdm_train.append(np.matrix(model.docvecs))
        logger.info("Using training vectors")
    else:
        trained_model = best_model_infer
        # infer training vectors on training data
        features_pvdm_train.append(infer_vectors(
            trained_model, read_corpus(folds_train_X[i]), 10, 0.0001, 0.1))
        logger.info("Using inferred training vectors")

    # infer training vectors on test data
    features_pvdm_train.append(infer_vectors(
        trained_model, read_corpus(folds_test_X[i]), 10, 0.0001, 0.1))


pickle.dump(features_pvdm_train,
            open("sentences_features_pvdm_train.pickle", "wb"))
pickle.dump(features_pvdm_test,
            open("sentences_features_pvdm_test.pickle", "wb"))

logger.info("Done")
