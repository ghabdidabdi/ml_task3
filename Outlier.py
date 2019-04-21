'''
adding outlier preprocessing
'''

# imports
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

#####################
### set up logger ###
#####################
import sys
sys.stdout = open('trainlog.log', 'w')

# read data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
index = test.index
features = train.drop(['y'], axis = 1)

##########################
### eliminate outliers ###
##########################

# create outlier classifier
outlier_clf = LocalOutlierFactor(n_neighbors = 21, # looking at 21 neighbors, since 21 is not divisible by 5
                                 contamination = 0.1 # how much data is assumed to be contaminated
)
print("training model for outliers")
y_pred = outlier_clf.fit_predict(features)
print("outliers detected, removing them")

# remove outliers
pca_in = []
y = []

for i in range(len(features)):
    if y_pred[i] == 1:
        y.append(train.y[i])
        pca_in.append(features[i])
print("outliers removed")

######################
### Normalize data ###
######################

print("normalizing data")
X = StandardScaler().fit_transform(pca_in)
test = StandardScaler().fit_transform(test)

##########################
### PCA- preprocessing ###
##########################

RANDOM = 42
DIM = 70

# creat input for PCA
pca_in = train.drop(['y'], axis = 1)

# create and fit PCA-model
print()
print()
print('Starting PCA for ', DIM, ' dimensions')
pca = PCA(n_components = DIM, random_state = RANDOM)

X = pca.fit_transform(X)
test = pca.transform(test)

print()
print()
print('PCA Done, ', pca.explained_variance_ratio_, ' explained')
print('sum: ', sum(pca.explained_variance_ratio_))

###########################
### tensorflow learning ###
###########################

import warnings
warnings.filterwarnings("ignore")

# import pandas as pd

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM)

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
    keras.layers.Dropout(0.4),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dropout(0.4),
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
    keras.layers.Dense(120, activation=tf.nn.tanh, input_dim=DIM),
    keras.layers.Dropout(0.01),
    keras.layers.Dense(240, activation=tf.nn.tanh),
    keras.layers.Dropout(0.01),
    keras.layers.Dense(120, activation=tf.nn.tanh),
    keras.layers.Dense(120, activation=tf.nn.tanh),
    keras.layers.Dropout(0.01),
    keras.layers.Dense(120, activation=tf.nn.tanh),
    keras.layers.Dense(120, activation=tf.nn.tanh),
    keras.layers.Dropout(0.01),
    keras.layers.Dense(60, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])
model4.name = 'model4'

models = [
    model1,
    model2,
    model3,
    model4,
]




results ={}

for model in models:
    print(model.name)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
#
#     model.fit(X_train, y_train, epochs = 50,
#               verbose = 2
#     )
#     results[model] = model.evaluate(X_test, y_test,
#               verbose = 2
#     )


# best_acc = 0.0

best_model = model2

# for result in results:
#     print(results.get(result))
#     if results.get(result)[1] > best_acc:
#         best_acc = results.get(result)[1]
#         best_model = result

# print('Best Model: ' + best_model.name + '; accuracy = ' + str(best_acc))

best_model.fit(X, y,  epochs = 75,
               verbose = 2
)
y_pred = best_model.predict_classes(test)

resf = pd.DataFrame({'Id': index, 'y': y_pred})
resf.to_csv('res_outlier.csv', index = False)
print('Done')
