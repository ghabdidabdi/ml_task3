'''
ML Task 3
'''

# imports
import pandas as pd
import tensorflow as tf
# from sklearn.neural_network import MLPClassifier as cls

# read data
train = pd.read_hdf("train.h5")
test = pd.read_hdf("train.h5")

# separate features from labels
features = train.drop(['y'], axis = 1)
labels = train.y

# build feature column
feature_column = []
keys = train.to_dict().keys()
for i in keys:
    feature_column.append(
        tf.feature_column.numeric_column(key=i)
    )

# create classifier
clf = tf.estimator.DNNClassifier(hidden_units = [120, 120, 120],
                                 feature_columns = feature_column
)

# train
clf.train(
    lambda: train.to_dict()
)
