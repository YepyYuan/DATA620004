### Full-connected Neural Network Image Classifier constructed by NumPy
This is a course project of _DATA620004, School of Data Science, Fudan University_.

##### Implementation
`Dataloader.py` : load Fashion-MNIST dataset and return `np.ndarray` of images and labels
`train.py`: train the model, save the model with highest validation accuracy as `best_model.pkl`, and return the lists of training loss, training accuracy, validation loss and validation accuracy
```l_t,a_t,l_v,a_v = train(model, images, labels, loss, optimizer, num_epochs, para_dic: dict)```
`test.py`: load a specified model and return the accuracy of test dataset
```test_accuracy=test(data_path, model_path)```

