import numpy as np

from base import Layer, Tensor

class Linear(Layer):
    """
    input X, shape: [N, C]
    output Y, shape: [N, O]
    weight W, shape: [C, O]
    bias b, shape: [1, O]
    grad dY, shape: [N, O]
    forward formula:
        Y = X @ W + b   # @表示矩阵乘法
    backward formula:
        dW = X.T @ dY
        db = sum(dY, axis=0)
        dX = dY @ W.T
    """
    def __init__(
        self,
        in_features,
        out_features,
        name='linear',
        
        *args,
        **kwargs
        ):
        
        super().__init__(name=name, *args, **kwargs)
        
        self.weights = Tensor((in_features, out_features))
        # self.weights.data = weight_attr(self.weights.data.shape)
        self.weights.data = np.random.normal(size= self.weights.data.shape)
        self.bias = Tensor((1, out_features))
        # self.bias.data = bias_attr(self.bias.data.shape)
        self.bias.data = np.zeros(shape= self.bias.data.shape)
        self.input = None

    def forward(self, x):
        self.input = x
        output = np.dot(x, self.weights.data) + self.bias.data
        return output

    def backward(self, gradient):
        self.weights.grad += np.dot(self.input.T, gradient)  # dy / dw
        self.bias.grad += np.sum(gradient, axis=0, keepdims=True)  # dy / db 
        input_grad = np.dot(gradient, self.weights.data.T)  # dy / dx
        return input_grad

    def parameters(self):
        return [self.weights, self.bias]

    def __str__(self):
        string = "linear layer, weight shape: {}, bias shape: {}".format(self.weights.data.shape, self.bias.data.shape)
        return string


class ReLU(Layer):
    """
    forward formula:
        relu = x if x >= 0
                = 0 if x < 0
    backwawrd formula:
        grad = gradient * (x > 0)
    """
    def __init__(self, name='relu', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.activated = None

    def forward(self, x):
        x[x < 0] = 0             
        self.activated = x
        return self.activated

    def backward(self, gradient):
        return gradient * (self.activated > 0) 