import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

'''
Trying outlier removing
'''

def drop_outliers(Iframe, ratio):
    '''drop given rate of outliers for labels 0..4'''
    # copy frame
    res = Iframe.copy()

    # split frame
    partitions = [Iframe[Iframe.y == i] for i in range(5)]
    lables = [[] for i in range(5)]

    # main loop
    for i in range(5):
        p = partitions[i]
        # train and predict
        clf = LocalOutlierFactor(contamination = ratio)
        y_pred = clf.fit_predict(p.drop(['y'], axis = 1).values)

        # insert in order to drop
        p.insert(0, column = 'outlier', value = y_pred)

        # now drop
        p = p[p.outlier != -1]
        partitions[i] = p.drop(['y'], axis = 1).values
        labels[i] = p.y

    # create v_stack for returning
    res = (np.vstack(partitions), np.vstack(labels))
    return res

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
index = test.index

X = train.drop(['y'], axis = 1).values
y = train.pop('y').values
test = test.values

X = StandardScaler().fit_transform(X)
test = StandardScaler().fit_transform(test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# list of ratios for contamination
c = [0.001, 0.01, 0.1, 0.15, 0.2]

results = {}
for cont in c:
    ## generate model
    model = keras.Sequential([
        keras.layers.Dense(240, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(240, activation=tf.nn.relu),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(240, activation=tf.nn.tanh),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(5, activation=tf.nn.softmax),
    ])
    model.name = 'model: contamination =  ' + str(cont)

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    ### drop outliers
    tmp = X_test.copy()
    tmp.insert(0, column = 'y', value = y_test)

    X_curr, y_curr = drop_outliers(tmp, cont)

    model.fit(X, y, epochs = 200)

    results[cont] = model.evaluate(X_test, y_test)
    print("done with ", model.name)
    print(results[cont])



# results ={}

# for model in models:

# # exit()

# best_acc = 0.0

# for result in results:
#     #print(results.get(result))
#     if results.get(result)[1] > best_acc:
#         best_acc = results.get(result)[1]
#         best_model = result

# print('Best Model: ' + best_model.name + '; accuracy = ' + str(best_acc))

# # best_model.fit(X, y,  epochs = 20)
# y_pred = best_model.predict_classes(test)

# resf = pd.DataFrame({'Id': index, 'y': y_pred})
# resf.to_csv('resTue_Apr_23_18:19:23_CEST_2019.csv', index = False)
# print('Done')
