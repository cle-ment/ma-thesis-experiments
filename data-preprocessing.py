# python
import datetime
import logging
import pickle

# packages
import numpy as np
import pandas as pd

# specific imports
import gensim


### Basic setup

TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")

## Logger setup

# create logger with 'joblearn'
logger = logging.getLogger('joblearn')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('logs/' + EXP_NAME + '-' + TIMESTAMP + '.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s:%(levelname)s: %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
# logger.addHandler(fh)
logger.addHandler(ch)


### Dataset Initialization

# read data
df = pd.read_csv("/Users/cwestrup/thesis/data/collected/"
                 + "crowdflower/data/sentences_aggregated_50-449.csv")

# Shuffle data
df = df.iloc[np.random.permutation(np.arange(len(df)))]

# Use entries with label confidence over 0.6 and aren't test questions:
df_conf = df[df['0_label:confidence'] > 0.6]
df_conf = df_conf[df_conf['_golden'] == False]
df_conf = df_conf[['0_label', '0_label:confidence', '0-sentence',
         '0-context-after', '0-context-before']]

pickle.dump(df_conf, open("data/sentences-dataframe-"
                          + "confidence_greater-0.6.pickle", "wb" ))
