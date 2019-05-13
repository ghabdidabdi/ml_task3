import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pandas as pd
import keras
from keras.utils import np_utils
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

# trainShape = train.shape
# testShape = test.shape

index = test.index

X = train.drop(['y'], axis = 1).values
y = train.pop('y').values
test = test.values

y = np_utils.to_categorical(y)

# idTest = np.arange(trainShape[0], trainShape[0]+testShape[0], dtype=np.int)

allData = np.vstack((X, test))

n_clusters = 100
pca = PCA(n_components=n_clusters, copy=False)
allData = pca.fit_transform(X=allData)
X = pca.transform(X=X)
test = pca.transform(X=test)

batch_size = 250

model = Sequential()
model.name = 'NetworkA'
model.add(Dense(500, input_shape=(n_clusters,), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, epochs=2000, verbose=1)

dataYPredict = model.predict(test)
y_pred = np.argmax(dataYPredict, axis=1)

resf = pd.DataFrame({'Id': index, 'y': y_pred})
resf.to_csv('res.csv', index = False)
print('Done')

