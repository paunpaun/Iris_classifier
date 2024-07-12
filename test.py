import pandas as pd
import numpy as np
from urllib.request import urlretrieve

iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

urlretrieve(iris)

df = pd.read_csv(iris, sep=',')

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

shuffled_indices = np.random.permutation(len(X))
X = X[shuffled_indices]
y = y[shuffled_indices]


print([[X,y] for X,y in zip(X,y)])