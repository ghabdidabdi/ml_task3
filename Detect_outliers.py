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

# separate different labels from eachother
sep = [train[train.y == i].drop(['y'], axis = 1) for i in range(5)]

# list of corruptions / neighbors to test for
tp = [
    (21, 0.1),
    (21, 0.2),
    (11, 0.1),
    (11, 0.2),
]

# main loop
for n, c in tp:
    res_tmp = [0] * len(train.y)
    for label in sep:
        # create outlier classifier
        label = label.copy()
        outlier_clf = LocalOutlierFactor(n_neighbors = n,
                                        contamination = c
        )
        print("training model for corruption: ", c, ', neighbors: ', n)
        y_pred = outlier_clf.fit_predict(label.values)

        # insert prediction into frame
        label.insert(0, column = 'outlier', value = y_pred)

        # iterate through prediction
        for p, ind in zip(y_pred, label.index):
            if p == -1:
                res_tmp[ind] = -1

    f = pd.DataFrame({'Out': res_tmp})
    filename = 'Outlier_multy_n={}_c={}.csv'.format(n, c)
    f.to_csv(filename)
