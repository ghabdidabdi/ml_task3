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
# import sys
# sys.stdout = open('trainlog.log', 'w')

##########################
### eliminate outliers ###
##########################

# define outlier configuration
N_NEIGHBORS = 21
CONTAMINATION = 0.1

# try reading a csv
y_pred = []
filename = 'Outlier_multy_n={}_c={}.csv'.format(N_NEIGHBORS, CONTAMINATION)
try:
    out_frame = pd.read_csv(filename)
    y_pred = out_frame.Out
except FileNotFoundError:
    # file was not found, create and train new model, then print results to csv
    print('file ', filename, ' was not found :(')
    print('new file will be generated')
    print()
    print('create new classifier')
    outlier_clf = LocalOutlierFactor(n_neighbors = N_NEIGHBORS,
                                    contamination = CONTAMINATION
    )
    print("training model for corruption: ", CONTAMINATION, ', neighbors: ', N_NEIGHBORS)
    y_pred = outlier_clf.fit_predict(features)
    print("outliers detected, creating csv")

    # create new frame and print it to csv
    f = pd.DataFrame({'Out': y_pred})
    f.to_csv(filename)

# read data
train_og = pd.read_hdf("train.h5", "train")
all_data = pd.read_hdf("train.h5", "train").drop(['y'], axis = 1)

# insert outlier-column
train_og.insert(0, column = 'outlier', value = y_pred)

# NOTE: split here
train_og, X_test = train_test_split(train_og, test_size=0.33)

# train_og = pd.read_hdf("train.h5", "train")
y_test = X_test.y
X_test = X_test.drop(['y', 'outlier'], axis = 1)

test = pd.read_hdf("test.h5", "test")
index = test.index

# remove outliers
train = train_og[train_og.outlier >= 0]
features = train.drop(['y', 'outlier'], axis = 1).values
y = train.pop('y')
print("outliers removed")

######################
### Normalize data ###
######################

pca_in = features
# print("normalizing data")
# pca_in = StandardScaler().fit_transform(features)
# test = StandardScaler().fit_transform(test)

# # testset
# X_test = StandardScaler().fit_transform(X_test)

##########################
### PCA- preprocessing ###
##########################

RANDOM = 42
DIM = 120

# create and fit PCA-model
print()
print()
print('Starting PCA for ', DIM, ' dimensions')
pca = PCA(n_components = DIM, random_state = RANDOM)

pca.fit(all_data.values)
# X = pca.fit_transform(pca_in)
# test = pca.transform(test)
X = pca_in

# testset
# X_test = pca.fit_transform(X_test)

print()
print()
# print('PCA Done, ', pca.explained_variance_ratio_, ' explained')
# print('sum: ', sum(pca.explained_variance_ratio_))

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
    # keras.layers.Dropout(0.1),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dense(120, activation=tf.nn.relu),
    # keras.layers.Dropout(0.1),
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

best_model.fit(X, y,  epochs = 150,
               verbose = 2
)

res = best_model.evaluate(X_test, y_test)

print('performance:')
print(res)

# NOTE: exit here, just for evaluation
exit()
y_pred = best_model.predict_classes(test)

resname = 'res_outlier_n={}_c={}.csv'.format(N_NEIGHBORS, CONTAMINATION)
resf = pd.DataFrame({'Id': index, 'y': y_pred})
resf.to_csv(resname, index = False)
print('Done')
