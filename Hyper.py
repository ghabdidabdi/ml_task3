# -*- flyspell-mode: nil -*-
# hyper: search over hyper-params
# local flag: train and evaluate locally, or train locally and evaluate online
Local = True
# logging flag: keep logging stuff
Logging = True

try: from tele_utils import quicksend, quicksend_file
except Exception:
    print('module \'tele_utils\' not available')
    def quicksend(*ars, **kwargs): return
    def quicksend_file(*ars, **kwargs): return

# create logger-file
import logging
from datetime import datetime as dt

# compute name for log-file
now = dt.now()
logfilename = 'logfile_hyper_{}_{}_{}_{}.log'.format(now.month, now.day, now.hour, now.minute)


# exit()
# standard imports
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# try auto-encoding with keras
from keras.layers import Input, Dense
from keras.models import Model

# if Logging is set to False: override LOGGER.info
def void(*args, **kwargs): pass
if not Logging:
    LOGGER.info = void
else:
    # otherwise setup logger
    logging.basicConfig(level=10, filename = logfilename)
    LOGGER = logging.getLogger()

def hyper(dim: int, depth: int, width: int):
    LOGGER.info('starting main for {} dimensions'.format(dim))

    # read data
    train = pd.read_hdf("train.h5", "train")
    test = pd.read_hdf("test.h5", "test")
    index = test.index

    ##################################################
    # # # preprocess data (scaling and encoding) # # #
    ##################################################
    X = train.drop(['y'], axis = 1).values
    y = train.pop('y').values
    test = test.values

    Scaler = StandardScaler()
    X = Scaler.fit_transform(X)
    test = Scaler.transform(test)

    enc_dim = dim # enc_dim: dimension to encode to

    # create input layer
    i_layer = Input(shape = (120,))

    # # create intermediate encoding layer
    # interm_dim = 120
    # interm_enc = Dense(interm_dim, activation = 'sigmoid')(i_layer)

    # create encoded layer
    e_layer = Dense(enc_dim, activation = 'relu')(i_layer)
    # e_layer = Dense(enc_dim, activation = 'sigmoid')(interm_enc)

    # # create intermediate decoding layer
    # interm_dec = Dense(interm_dim, activation = 'sigmoid')(e_layer)

    # create decoded layer
    d_layer = Dense(120)(e_layer)
    # d_layer = Dense(120)(interm_dec)

    # create auto-encoder, the model that maps input straight to output
    auto_encoder = Model(i_layer, d_layer)

    # encoder: map input to lower dimension
    encoder = Model(i_layer, e_layer)

    # # create model for decoding
    # enc_input = Input(shape = (enc_dim,))
    # dec_layer = auto_encoder.layers[-1]
    # decoder = Model(enc_input, dec_layer(enc_input))

    # now let's train!
    # NOTE: we encode our entire X!
    auto_encoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
    auto_encoder.fit(X, X,
                    epochs = 75
    )

    # and now we can encode our data:
    X = encoder.predict(X)
    test = encoder.predict(test)

    # update user
    print('encoding done!')

    # now we do regular prediction on encoded data, if the local-flag is set to True
    X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.33, random_state=11312)
                                        if Local
                                        else (X, y, test, test))


    ################################################
    # # # create model from given hyper-params # # #
    ################################################
    # TODO
    # array of layers, first layer takes enc_dim inputs
    layerz = [keras.layers.Dense(dim, activation=tf.nn.relu, input_dim = enc_dim)]
    for i in range(1, depth):
        layerz.append(keras.layers.Dense(dim, activation=tf.nn.relu, input_dim = dim))
    # append last layer, that only outputs 5 weights
    layerz.append(keras.layers.Dense(5, activation=tf.nn.softmax, input_dim = dim))

    # Model 2 from earlier
    model = keras.Sequential(layerz)
    model.name = 'model_hyper_{}_{}_{}'.format(dim, depth, width)

    model.compile(
        optimizer='adam'
        , loss='categorical_crossentropy'
        # , loss='sparse_categorical_crossentropy'
        # NOTE: is this the correct metrics?
        , metrics=['accuracy']
    )
    #
    model.fit(X_train, y_train, epochs = 200)

    LOGGER.info('Done, {} now'.format('evaluating' if Local else 'predicting'))

    if Local:
        # if local is set to True: evluate on local data
        results = model.evaluate(X_test, y_test)
        print("done :D")
        quicksend("done :D, for: {}, {}, {}:".format(dim, depth, width))
        print(results)
        quicksend(results)

        LOGGER.info('evaluation done, results:')
        LOGGER.info(results)
        now = dt.now()
        LOGGER.info('timestamp: ' + '{}, {}, {}\n\n'.format(now.month, now.day, now.hour, now.minute))
        return results
    else:
        # otherwise predict test-set and print to csv
        y_pred = model.predict_classes(test)
        resf = pd.DataFrame({'Id': index, 'y': y_pred})

        # get the filename:
        # import datetime.datetime as dt
        now = dt.now()
        filename = 'Results_task3_{}_{}_{}_{}.csv'.format(now.month, now.day, now.hour, now.minute)
        resf.to_csv(filename, index = False)
        print('Done')

def search():
    '''search over our hyperparameters for something maybe usefull'''
    # generate ranges for our hyper-params
    r_dim = list(range(40, 121, 10))
    r_width = list(range(40, 481, 40))
    r_depth = list(range(1, 10))

    # generate permutations
    cross = [
        (i, j, k) for i in r_dim
        for j in r_width
        for k in r_depth
    ]

    # and run hyper for all permutations, store all the results in db
    import sqlite3
    hyper_db = pd.DataFrame(columns = ['dim', 'width', 'depth', 'acc', 'loss'])
    for (di, wi, de) in cross:
        # wrapped in try just to make sure
        try:
            loss, acc = hyper(di, wi, de)
            # open db
            conn = sqlite3.connect('hyper.db')
            c = conn.cursor()
            # insert results
            query = 'INSERT INTO performances VALUES ({}, {}, {}, {}, {});'.format(di, wi, de, loss, acc)
            c.execute(query)
            # close connection
            conn.commit()
            conn.close()
        except Exception:
            print('error for params: {}, {}, {}'.format(di, wi, de))
            quicksend('error for params: {}, {}, {}'.format(di, wi, de))

if __name__ == '__main__':
    quicksend('starting now, in')
    quicksend('3')
    quicksend('2')
    quicksend('1')
    search()
    LOGGER.info('all done')
    quicksend('all done')
