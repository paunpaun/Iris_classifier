import numpy as np
import pandas as pd
from functions import Sigmoid, Mse, Leaky_ReLU, CategoricalCrossEntropy, Softmax, ReLU

class Layer:
    def __init__(self, n_inputs, n_neurons,act_f):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons).reshape(1,-1)
        self.w_grads = 0
        self.inputs = 0
        self.act_f = act_f
        self.grad = 0
        self.z = 0
    
    def __call__(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        output = self.act_f(self.z) 
        return output
    
    def backwards(self, chain):
        chain = np.array(chain).reshape(1,-1)
        self.grad = chain * self.act_f.derivative(self.z)
        self.w_grads = self.inputs.T @ self.grad

    def update(self):
        self.grad = 0
        self.weights += -lr * self.w_grads
        self.biases += -lr * np.sum(self.grad,axis=0, keepdims=True)

class MLP:

    def __init__(self, n_inputs, n_neurons,act_f):
        size = [n_inputs] + n_neurons
        n_layers = len(n_neurons)
        self.layers = [Layer(size[i], size[i+1],act_f[i]) for i in range(n_layers)]

    def __call__(self, inputs):
        for i in range(len(self.layers)):
            inputs = self.layers[i](inputs)
        return inputs
    
    def backwards(self, d_loss):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[-1].backwards(d_loss)
                continue

            chain = self.layers[-i].grad @ self.layers[-i].weights.T
            self.layers[-1-i].backwards(chain)

    def update(self):
        for layer in self.layers:
            layer.update()

cce = CategoricalCrossEntropy()
leaky_relu = Leaky_ReLU()
sigmoid = Sigmoid()
mse = Mse()
softmax = Softmax()
relu = ReLU()


lrs = np.linspace(0 , -3 ,num=1000)

mlp = MLP(4,[3], [softmax])

from urllib.request import urlretrieve

iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

urlretrieve(iris)

df = pd.read_csv(iris, sep=',')

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

split_index = int(0.8 * len(X))
shuffled_indices = np.random.permutation(len(X))
X = X[shuffled_indices]
y = y[shuffled_indices]

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

#-----------------------------------------------------
counter1 = 0
counter2 = 0
counter3 = 0
loss_values = []
loss_values2 = []
output_values = []
target_values = []

for _ in range(1000):
    
    lr = 0.000005
    

    i = np.random.randint(1, len(X_train)) 

    target = y_train[i]
    if target == 'Iris-setosa': 
        target = [1,0,0]
        counter1 += 1
    if target == 'Iris-versicolor':
        target = [0,1,0]
        counter2 += 1
    if target == 'Iris-virginica': 
        target = [0,0,1]
        counter3 += 1
    inputs = X_train[i].reshape(1,-1)
    outputs = mlp(inputs)[0]

    loss = cce(outputs,target)

    output_values.append(outputs)
    target_values.append(target)
    loss_values.append(loss)

    d_loss = cce.derivative(outputs, target)

    mlp.backwards(d_loss)
    mlp.update()

    print(f'output: {outputs}, target: {target}, loss: {loss}, d_loss: {d_loss}')

print('======================================')

for i in range(len(X_test)):

    target = y_test[i]
    if target == 'Iris-setosa': target = [1,0,0]
    if target == 'Iris-versicolor': target = [0,1,0]
    if target == 'Iris-virginica': target = [0,0,1]
    inputs = X_test[i]
    outputs = mlp(inputs)[0]
    

    loss = cce(outputs,target)
    loss_values2.append(loss)

    print(f'output: {outputs}, target: {target}, loss = {loss}')

import matplotlib.pyplot as plt
 
x = np.arange(len(output_values))
y1 = output_values
y2 = target_values
y3 = loss_values


print(f'c1: {counter1}, c2: {counter2}, c3: {counter3}')
plt.plot(x, y3, 'o',color='blue')

plt.show()

x = np.arange(len(loss_values2))
y1 = loss_values2


plt.plot(x, y1, 'o',color='blue')

plt.show()
