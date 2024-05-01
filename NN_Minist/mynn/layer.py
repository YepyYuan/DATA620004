import numpy as np
from .base import Layer, Tensor

class Linear(Layer):
    """
    forward formula:
        Y = X @ W + b   @ is matrix multiplication
    backward formula:
        dW = X.T @ dY / X.shape[0]
        db = mean(dY, axis=0)
        dX = dY @ W.T
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        name='linear',
        *args,
        **kwargs
        ):
        
        super().__init__(name=name, *args, **kwargs)
        
        self.weights = Tensor((in_features, out_features))
        self.weights.data = np.random.randn(in_features, out_features)
        self.bias = Tensor((1, out_features))
        self.bias.data = np.random.randn(1, out_features)
        self.input = None

    def forward(self, x):
        self.input = x.copy()
        output = np.dot(x, self.weights.data) + self.bias.data
        return output

    def backward(self, gradient):
        self.weights.grad = np.dot(self.input.T, gradient) / self.input.shape[0]
        self.bias.grad = gradient.mean(axis=0)
        # return input_grad
        return np.einsum('ij,kj->ik', gradient, self.weights.data, optimize=True)

    def parameters(self):
        return [self.weights, self.bias]


class ReLU(Layer):
    """
    forward formula:
        y = x if x >= 0
          = 0 if x < 0
    backwawrd formula:
        grad = gradient * (x > 0)
    """
    def __init__(self, name='relu', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.activated = None

    def forward(self, x: np.ndarray):
        x[x < 0] = 0             
        self.activated = x
        return self.activated

    def backward(self, gradient: np.ndarray):
        return gradient * (self.activated > 0) 
    
class SoftMax(Layer):
    """
    forward fomula:
        y = exp(x) / sum(exp(x))
    backward formula:
        grad = gradient of cross entropy
    """

    def __init__(self, name='softmax', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.activated = None

    def forward(self, x: np.ndarray):
        x = x - x.max(axis=-1, keepdims=True)
        self.output = np.divide(np.exp(x), np.exp(x).sum(axis=-1)[:, None])
        return self.output
    
    def backward(self, gradient: np.ndarray):
        return gradient
   
class CrossEntropy(Layer):
    """
    forward formula:
        y = output of softmax
    backward formula:
        grad = yi[label] - 1
    """
    def __init__(self, model, name='ce', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.model = model
        self.activated = None

    def forward(self, pred: np.ndarray, label: np.ndarray):
        self.batch_size, self.label_size = pred.shape
        self.label = label
        self.output = pred
        return self
    
    def backward(self):
        gradient = self.output.copy()
        gradient[range(self.batch_size), self.label] -= 1
        # self.model.backward(gradient / self.label_size)
        self.model.backward(gradient)
        # return gradient
    
    def item(self):
        return - np.log(self.output[np.arange(self.batch_size), self.label] + 1e-7).mean()