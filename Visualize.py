'''
visualize stuff lol
'''

##########################
### PCA- preprocessing ###
##########################

RANDOM = 42
DIM = 2

# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# read data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
index = test.index

# creat input for PCA
pca_in = train.drop(['y'], axis = 1)

# create and fit PCA-model
print()
print()
print('Starting PCA for ', DIM, ' dimensions')
pca = PCA(n_components = DIM, random_state = RANDOM)

X = pca.fit_transform(pca_in)
y = train.pop('y').values
test = pca.transform(test)

# PCA done
print()
print()
print('PCA Done, ', pca.explained_variance_ratio_, ' explained')
print('sum: ', sum(pca.explained_variance_ratio_))

# plot
plt.scatter(X[:, 0], X[:, 1])
plt.show()
