from Dataloader import load_mnist
from mynn.base import Sequential
from mynn.layer import Linear, ReLU, SoftMax, CrossEntropy
from mynn.optimize import SGD
from train import train, load_hyper_para
import numpy as np

# load data
path = './data'
image, label = load_mnist(path, kind= 'train')
image= image.copy().astype(np.float64)
image /= 255

## hyper parameter list
hidden_layer_list = [384, 256, [384, 128], [512,256]]
lr_list = [0.05, 0.01, 0.005 ]
decay_list = [0, 0.05, 0.1]

def main():

    for hidden_layer in hidden_layer_list:
        for lr in lr_list:
            for decay in decay_list:
                
                model, loss_fun, optimizer, para_dic = load_hyper_para(hidden_layer, lr, decay)
                train_loss, train_acc, val_loss, val_acc = train(model, image, label, loss_fun, optimizer, para_dic=para_dic, num_epochs=5, iter_valid=50)


if __name__ == '__main__':
    main()