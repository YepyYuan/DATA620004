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

    def __str__(self):
        return self.name
    

class Optimizer:
    """
    optimizer base class.
    Args:
        parameters (Tensor): parameters to be optimized.
        learning_rate (float): learning rate. Default: 0.001.
        weight_decay (float): The decay weight of parameters. Defaylt: 0.0.
        decay_type (str): The type of regularizer. Default: l2.
    """
    def __init__(self, parameters, learning_rate=0.001, weight_decay=0.0, decay_type='l2'):
        assert decay_type in ['l1', 'l2'], "only support decay_type 'l1' and 'l2', but got {}.".format(decay_type)
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_type = decay_type
        
    def step(self):
        raise NotImplementedError

    def clear_grad(self):
        for p in self.parameters:
            p.clear_grad()

    def get_decay(self, g):
        if self.decay_type == 'l1':
            return self.weight_decay
        elif self.decay_type == 'l2':
            return self.weight_decay * g
    

