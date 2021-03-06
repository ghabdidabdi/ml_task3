# local flag: train and evaluate locally, or train locally and evaluate online
Local = True
Logging = False

# create logger-file
import logging
from datetime import datetime as dt

# compute name for log-file
now = dt.now()
logfilename = 'logfile_autoenc_{}_{}_{}_{}.log'.format(now.month, now.day, now.hour, now.minute)

# setup logger
logging.basicConfig(level=10, filename = logfilename)
LOGGER = logging.getLogger()

# if Logging is set to False: override LOGGER.info
def void(*args, **kwargs): pass
if not Logging:
    LOGGER.info = void

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

def main(dim = 84):
    global Local, LOGGER

    LOGGER.info('starting main for {} dimensions'.format(dim))

    # read data
    train = pd.read_hdf("train.h5", "train")
    test = pd.read_hdf("test.h5", "test")
    index = test.index

    X = train.drop(['y'], axis = 1).values
    y = train.pop('y').values
    test = test.values

    X = StandardScaler().fit_transform(X)
    test = StandardScaler().fit_transform(test)

    enc_dim = dim # enc_dim: dimension to encode to

    # create input layer
    i_layer = Input(shape = (120,))

    # create intermediate encoding layer
    interm_dim = 480
    interm_enc = Dense(interm_dim, activation = 'sigmoid')(i_layer)

    # create encoded layer
    # e_layer = Dense(enc_dim, activation = 'relu')(i_layer)
    e_layer = Dense(enc_dim, activation = 'sigmoid')(interm_enc)

    # create intermediate decoding layer
    interm_dec = Dense(interm_dim, activation = 'sigmoid')(e_layer)

    # create decoded layer
    d_layer = Dense(120)(interm_dec)

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
                    epochs = 500
    )

    # and now we can encode our data:
    X = encoder.predict(X)
    test = encoder.predict(test)

    # update user
    print('encoding done!')

    # now we do regular prediction on encoded data, if the local-flag is set to True
    X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.33, random_state=42)
                                        if Local
                                        else (X, y, test, test))

    # Model 2 from earlier
    model = keras.Sequential([
        keras.layers.Dense(120, activation=tf.nn.relu, input_dim=enc_dim),
        keras.layers.Dense(120, activation=tf.nn.tanh),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(120, activation=tf.nn.relu),
        keras.layers.Dense(120, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation=tf.nn.softmax)
    ])
    model.name = 'model2'

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #
    model.fit(X_train, y_train, epochs = 1000)

    LOGGER.info('Done, {} now'.format('evaluating' if Local else 'predicting'))

    if Local:
        # if local is set to True: evluate on local data
        results = model.evaluate(X_test, y_test)
        print("done :D")
        print(results)

        LOGGER.info('evaluation done, results:')
        LOGGER.info(results)
        now = dt.now()
        LOGGER.info('timestamp: ' + '{}, {}, {}\n\n'.format(now.month, now.day, now.hour, now.minute))
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

if __name__ == '__main__':
    main(84)
    # for i in range(20, 100, 2):
    #     main(i)
