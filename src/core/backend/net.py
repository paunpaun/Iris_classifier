import random

class Neuron:

    def __init__(self, n_inputs, act_f):
        self.weights = [random.uniform(-1,1) for _ in range (n_inputs)]
        self.bias = random.uniform(-1,1)
        self.grad = 0
        self.z = 0
        self.inputs = 0
        self.act_f = act_f
        self.w_grads = [0 for _ in range(n_inputs)]

    def __call__(self, inputs):
        self.inputs = inputs
        self.z = sum(wi*xi for wi, xi in zip(self.weights, inputs)) + self.bias
        output = self.act_f(self.z)
        return output

    def parameters(self):
        return self.weights + [self.bias]
    
    def backwards(self,d_loss):
        self.grad = d_loss*self.act_f.derivative(self.z)
        for i in range(len(self.w_grads)):
            self.w_grads[i] = self.grad*self.inputs[i]

    def update(self,lr):
        for i in range(len(self.w_grads)):
            self.weights[i] += -lr * self.w_grads[i]
        self.bias +=  -lr * self.grad

class Layer:

    def __init__(self, n_inputs, n_neurons, act_f):
        self.neurons = [Neuron(n_inputs, act_f) for _ in range(n_neurons)]

    def __call__(self, inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def backwards(self,d_loss):
        print('backwards')
        for i, neuron in enumerate(self.neurons):
            neuron.backwards(d_loss[i])

    def update(self,lr):
        for neuron in self.neurons:
            neuron.update(lr)

class MLP:
    def __init__(self, n_inputs, n_neurons, act_f):
        size = [n_inputs] + n_neurons
        n_layers = len(n_neurons)
        self.layers = [Layer(size[i], size[i+1], act_f) for i in range(n_layers)]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def backwards(self,d_loss):
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                layer.backwards(d_loss)
                continue

            d_loss = []

            for j in range(len(layer.neurons)):
                tmp_list = []
                for neuron in self.layers[len(self.layers) - i].neurons:
                    g = neuron.grad
                    w = neuron.weights[j]
                    tmp_list.append(g*w)
                d_loss.append(sum(tmp_list))

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

