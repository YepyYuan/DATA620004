class SGD(object):
    """
    theta_{t+1} = (1- lambda) * theta_{t} - lr * grad
    lambda is the regularization coefficient
    """
    def __init__(self, model, lr=0.001, decay=None):
        self.lr = lr
        self.decay = decay
        self.model = model
        self.cur_step = 0
        self.parameters = self.model.parameters()

    def step(self):
        for para in self.parameters:
            grad = para.grad.copy()
            if self.decay:
                para.data -= para.data * self.decay * self.lr
            para.data -= self.lr * grad

        self.cur_step += 1

    def zero_grad(self):
        for para in self.parameters:
            para.clear_grad()