import numpy as np
import pandas as pd
from core.backend.neural_net import MLP
from core.backend.functions import Sigmoid, ReLU, Leaky_ReLU, Softmax, CategoricalCrossEntropy, Mse 


# activation functions
sigmoid = Sigmoid()
relu = ReLU()
leaky_relu = Leaky_ReLU()
softmax = Softmax()

# loss functions
cce = CategoricalCrossEntropy()
mse = Mse()

# data loading
iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(iris_url, header=None, names=attributes)

# data prep
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# one-hot encoding
y = pd.get_dummies(y).values

# split data
split_index = int(0.8 * len(X))
indices = np.random.permutation(len(X))
X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

# hyperparameters
lr = 0.01
iterations = 1000

# MLP
mlp = MLP(4, [4, 3], [relu, softmax])

losses = []

# training loop
for i in range(iterations):
    
    idx = np.random.randint(len(X_train))
    inputs, target = X_train[idx].reshape(1, -1), y_train[idx].reshape(1, -1)
    
    # forward pass
    outputs = mlp(inputs)

    # compute loss
    loss = cce(outputs, target)
    losses.append(loss)

    # backward pass
    d_loss = cce.derivative(outputs, target)
    mlp.backwards(d_loss)

    # update weights
    mlp.update(lr)
    
    if i % 100 == 0:
        print(f'Iteration {i}, Loss: {loss}')

correct = 0
for inputs, target in zip(X_test, y_test):
    outputs = mlp(inputs.reshape(1, -1))
    if np.argmax(outputs) == np.argmax(target):
        correct += 1

accuracy = correct / len(X_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
 
# plotting loss
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()