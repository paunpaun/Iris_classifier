import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, act_f):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.random.randn(n_neurons).reshape(1,-1)
        self.act_f = act_f
        self.inputs = None
        self.z = None
        self.grad = None
        self.w_grads = None
        self.b_grads = None
    
    def __call__(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs , self.weights) + self.biases
        return self.act_f(self.z) 
    
    def backwards(self, chain):
        self.grad = chain * self.act_f.derivative(self.z)
        self.w_grads = np.dot(self.inputs.T, self.grad)
        self.b_grads = np.sum(self.grad, axis=0, keepdims=True)

    def update(self, lr):
        self.weights -= lr * self.w_grads
        self.biases -= lr * self.b_grads

class MLP:

    def __init__(self, n_inputs, n_neurons, act_f):
        list_layers = [n_inputs] + n_neurons
        self.layers = [Layer(list_layers[i], list_layers[i+1],act_f[i]) for i in range(len(n_neurons))]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def backwards(self, d_loss):
        chain = d_loss
        for layer in reversed(self.layers):
            layer.backwards(chain)
            chain = np.dot(layer.grad, layer.weights.T)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)
