import numpy as np

class Tensor:

    def __init__(self, shape, requires_grad: bool = True):
        
        self.data = np.zeros(shape= shape, dtype=np.float32)
        if requires_grad:
            self.grad = np.zeros(shape= shape, dtype=np.float32)
        else:
            self.grad = None

    def clear_grad(self):
        self.grad = np.zeros_like(self.grad)

class Layer:
    def __init__(self, name='layer', *args, **kwargs):
        self.name = name

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    

class Sequential:
    def __init__(self, *args, **kwargs):
        self.layers = []
        self._parameters = []
        for arg_layer in args:
            if isinstance(arg_layer, Layer):
                self.layers.append(arg_layer)
                self._parameters += arg_layer.parameters()
 
    def add(self, layer):
        assert isinstance(layer, Layer), "The type of added layer must be Layer, but got {}.".format(type(layer))
        self.layers.append(layer)
        self._parameters += layer.parameters()
 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
 
    def backward(self, grad):
        # grad backward in inverse order of graph
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
 
    def parameters(self):
        return self._parameters
     
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)