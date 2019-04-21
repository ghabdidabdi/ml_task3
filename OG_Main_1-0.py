import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
index = test.index

X = train.drop(['y'], axis = 1).values
y = train.pop('y').values
test = test.values

X = StandardScaler().fit_transform(X)
test = StandardScaler().fit_transform(test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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
    keras.layers.Dense(120, activation=tf.nn.relu, input_dim=120),
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
    keras.layers.Dense(120, activation=tf.nn.relu, input_dim=120),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(60, activation=tf.nn.relu),
   keras.layers.Dense(5, activation=tf.nn.softmax)
])
model3.name = 'model3'

models = [model1, model2, model3]




results ={}

for model in models:
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
#
    model.fit(X_train, y_train, epochs = 5)
    results[model] = model.evaluate(X_test, y_test)


best_acc = 0.0

for result in results:
    #print(results.get(result))
    if results.get(result)[1] > best_acc:
        best_acc = results.get(result)[1]
        best_model = result

print('Best Model: ' + best_model.name + '; accuracy = ' + str(best_acc))

best_model.fit(X, y,  epochs = 20)
y_pred = best_model.predict_classes(test)

resf = pd.DataFrame({'Id': index, 'y': y_pred})
resf.to_csv('res.csv', index = False)
print('Done')