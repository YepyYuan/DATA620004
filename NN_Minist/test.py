<<<<<<< HEAD
import pickle
from Dataloader import load_mnist
import numpy as np

def test(data_path, model_path, load_kind = 't10k'):

    image , label = load_mnist(data_path, kind= load_kind)
    image= image.copy().astype(np.float64)

    ## Data Normalization
    image /= 255

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    pred = model(image).argmax(axis = 1)

    test_accuracy = (pred == label).mean()

    return test_accuracy
=======
from Dataloader import load_mnist

path = '.data'

image, label = load_mnist(path, kind= 'train')
>>>>>>> 54430f702e422c9bc66ee0ce6c889821f15a8cc7
