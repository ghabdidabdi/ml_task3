# -*- flyspell-mode: nil -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import pandas as pd
import keras
from keras.utils import np_utils
from sklearn.decomposition import PCA
# local testing
from sklearn.model_selection import train_test_split

try: from tele_utils import quicksend, quicksend_file
except Exception: pass

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

X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.33, random_state=11312))
# X_train, y_train = X, y

batch_size = 250

model = Sequential()
model.name = 'NetworkA'
model.add(Dense(500, input_shape=(n_clusters,), activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, epochs=1000, verbose=1)
# model.fit(X_train, y_train, batch_size=batch_size, epochs=1000, verbose=1)

# perf = model.evaluate(X_test, y_test)
# print('done')
# print(perf)

# try:
#     quicksend('done')
#     quicksend(perf)
# except Exception: pass

dataYPredict = model.predict(test)
y_pred = np.argmax(dataYPredict, axis=1)

resf = pd.DataFrame({'Id': index, 'y': y_pred})
resf.to_csv('res_hopefully_last_one_like_srsly.csv', index = False)
print('Done')
try: quicksend('done now, yayyyy!!!')
except Exception: pass
