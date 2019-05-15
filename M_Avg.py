import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import tensorflow as tf
import keras
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try: from tele_utils import quicksend
except Exception: pass

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
index = test.index

X = train.drop(['y'], axis = 1).values
y = train.pop('y').values
#y = np_utils.to_categorical(y)
test = test.values

X = StandardScaler().fit_transform(X)
test = StandardScaler().fit_transform(test)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## model 1
model1 = keras.Sequential([
    keras.layers.Dense(240, activation=tf.nn.relu,input_shape=(120,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(240, activation=tf.nn.tanh),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])
model1.name = 'model1'

## model 2

model2 = keras.Sequential([
    keras.layers.Dense(240, activation=tf.nn.relu, input_dim=120),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(480, activation=tf.nn.tanh),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
   keras.layers.Dense(5, activation=tf.nn.softmax)
])
model2.name = 'model2'

## model 3

model3 = keras.Sequential([
    keras.layers.Dense(120, activation=tf.nn.relu, input_dim=120),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
   keras.layers.Dense(5, activation=tf.nn.softmax)
])
model3.name = 'model3'

## model 4

model4 = keras.Sequential([
    keras.layers.Dense(240, activation=tf.nn.relu, input_dim=120),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
   keras.layers.Dense(5, activation=tf.nn.softmax)
])
model4.name = 'model4'

models = [model1, model2, model3, model4]

results ={}
predictions = {}
# compiling loop
for model in models:
    model.compile(optimizer='Adadelta',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

for i in range(1, 11):
    # update vars to save new predictions
    predictions[i] = []
    results[i] = {}

    # train 100 epochs for every model
    # then make predictions
    for model in models:
        model.fit(X, y, epochs = 100, batch_size=120, verbose=1)
        results[i][model] = model.predict(test)
        if predictions[i] == []:
            predictions[i] = results[i][model]
        else:
            predictions[i] += results[i][model]

    # now get model_avg prediction, print them to csv
    y_pred = np.argmax(predictions[i], axis=1)

    resf = pd.DataFrame({'Id': index, 'y': y_pred})
    resf.to_csv('res_model_avg_{}_iter.csv'.format(i * 100), index = False)
    print('Done with {} iterations'.format(i * 100))
    try: quicksend('Done with {} iterations'.format(i * 100))
    except Exception: pass
