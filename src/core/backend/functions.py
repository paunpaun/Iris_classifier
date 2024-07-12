import numpy as np

class Leaky_ReLU:

    def __call__(self, x):
        return np.maximum(0.01*x, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0.01)

class Sigmoid:

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        output = self(x) 
        return output * (1 - output)
    
class Mse:
    def __call__(self,predictions,targets):
        return 1/len(predictions)*sum([(p - t)**2 for p,t in zip(predictions,targets)])


    def derivative(self,predictions,targets):
        out = [2*(p - t) for p,t in zip(predictions,targets)]
        return [item*1/len(predictions) for item in out]
     
class BinarCrossEntropy:
    def binary_crossentropy(y_true, y_pred):
        epsilon = 1e-15  
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

class CategoricalCrossEntropy:
    def __call__(self,predictions,targets):
        return -np.sum(targets * np.log(predictions))  
    
    def derivative(self,predictions, targets):
        return predictions - targets

class Softmax:
    def __call__(self,x):
        return np.exp(x) / np.sum(np.exp(x))

    def derivative(self, x):
        return 1
    
class ReLU:
    def __call__(self,x):
        return np.maximum(0,x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)