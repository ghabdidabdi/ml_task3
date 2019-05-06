'''
trying out PCA from sklearn
'''

#####################
### set up logger ###
#####################
# import sys
# sys.stdout = open('trainlog.log', 'w')
# import logging

# # conf. and creat logger
# logging.basicConfig(filename = 'saumah.log',
#                     level = logging.DEBUG
# )
# m_log = logging.getLogger()

##########################
### PCA- preprocessing ###
##########################

RANDOM = 42
DIM = 40

# imports
import pandas as pd
from sklearn.decomposition import KernelPCA
import tensorflow as tf
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# read data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
index = test.index

# creat input for PCA
X = train.drop(['y'], axis = 1)
y = train.pop('y').values

# split data
X_pca, nothing, whatever, donotcare = train_test_split(X, y, test_size=0.9, random_state=RANDOM)

# create and fit PCA-model
print()
print()

print('Starting PCA for ', DIM, ' dimensions')
pca = KernelPCA(n_components = DIM, random_state = RANDOM, kernel = 'rbf')


# apply transformation
pca.fit(X_pca)
X = pca.transform(X)
print('fitting done :D')

test = pca.transform(test)
print('data transformed')

# f = pd.DataFrame({'Dim': list(range(1, 121)), 'ratio': dims})
# f.to_csv('ratio.csv', index = False)
# exit()
# PCA done

# print()
# print()
# print('PCA Done, ', pca.explained_variance_ratio_, ' explained')
# print('sum: ', sum(pca.explained_variance_ratio_))

###########################
### tensorflow learning ###
###########################

import warnings
warnings.filterwarnings("ignore")

# import pandas as pd

X = StandardScaler().fit_transform(X)
test = StandardScaler().fit_transform(test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM)

## model 1
model1 = keras.Sequential([
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dense(120, activation=tf.nn.tanh),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(60, activation=tf.nn.relu),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])
model1.name = 'model1'

## model 2

model2 = keras.Sequential([
    keras.layers.Dense(120, activation=tf.nn.relu, input_dim=DIM),
    keras.layers.Dense(120, activation=tf.nn.tanh),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])
model2.name = 'model2'

## model 3

model3 = keras.Sequential([
    keras.layers.Dense(120, activation=tf.nn.relu, input_dim=DIM),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(60, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])
model3.name = 'model3'

## model 4
model4 = keras.Sequential([
    keras.layers.Dense(120, activation=tf.nn.relu6, input_dim=DIM),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(240, activation=tf.nn.relu6),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(120, activation=tf.nn.relu6),
    keras.layers.Dense(120, activation=tf.nn.relu6),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(120, activation=tf.nn.relu6),
    keras.layers.Dense(120, activation=tf.nn.relu6),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(60, activation=tf.nn.relu6),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])
model4.name = 'model4'

models = [
    model2,
    model4,
]




results ={}

for model in models:
    print(model.name)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #
    model.fit(X_train, y_train, epochs = 50,
              verbose = 2
    )
    results[model] = model.evaluate(X_test, y_test,
              verbose = 2
    )
    print(model.name, ':')
    print(results[model])


# best_acc = 0.0

# for result in results:
#     print(results.get(result))
#     if results.get(result)[1] > best_acc:
#         best_acc = results.get(result)[1]
#         best_model = result

# print('Best Model: ' + best_model.name + '; accuracy = ' + str(best_acc))

# best_model.fit(X, y,  epochs = 200,
#                verbose = 2
# )
# y_pred = best_model.predict_classes(test)

# resf = pd.DataFrame({'Id': index, 'y': y_pred})
# resf.to_csv('res_pca.csv', index = False)
# print('Done')
