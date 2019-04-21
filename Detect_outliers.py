'''
Detects outliers for given 'contamination' and 'neighbors', prints to csv
'''

# imports
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# read data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
index = test.index
features = train.drop(['y'], axis = 1).values

# list of corruptions / neighbors to test for
tp = [
    (21, 0.2),
    (21, 0.1),
    (21, 0.01),
    (11, 0.2),
    (11, 0.1),
    (11, 0.01),
]

# main loop
for n, c in tp:
    # create outlier classifier
    outlier_clf = LocalOutlierFactor(n_neighbors = n,
                                    contamination = c
    )
    print("training model for corruption: ", c, ', neighbors: ', n)
    y_pred = outlier_clf.fit_predict(features)
    print("outliers detected, creating csv")

    f = pd.DataFrame({'Out': y_pred})
    filename = 'Outlier_n={}_c={}.csv'.format(n, c)
    f.to_csv(filename)
