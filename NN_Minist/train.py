from tqdm import tqdm
import pickle
import os
from datetime import datetime
from mynn.base import Sequential
from mynn.layer import Linear, ReLU, SoftMax, CrossEntropy
from mynn.optimize import SGD

def load_hyper_para(hidden_layer: int | list , lr = 0.01, decay = None):
    if isinstance(hidden_layer, int):
        model = Sequential(
            Linear(in_features=784, out_features= hidden_layer, name='linear1'),
            ReLU(name= 'relu1'),
            Linear(in_features= hidden_layer, out_features= 10, name='output'),
            SoftMax(name='sf')
        )
    else:
        assert isinstance(hidden_layer, list), "The type of hidden layer must be List, but got {}.".format(type(hidden_layer))
        model = Sequential()
        for k in range(len(hidden_layer)+1):
            if k == 0:
                model.add(Linear(in_features=784, out_features= hidden_layer[k], name='linear%d' %(k+1)))
                model.add(ReLU(name='relu%d' %(k+1)))
            elif k < len(hidden_layer):
                model.add(Linear(in_features=hidden_layer[k-1], out_features= hidden_layer[k], name='linear%d' %(k+1)))
                model.add(ReLU(name='relu%d' %(k+1)))
            else:
                model.add(Linear(in_features=hidden_layer[k-1], out_features= 10, name='output'))
                model.add(SoftMax(name='sf'))

    optimizer = SGD(model, lr=lr, decay=decay)
    loss_fun = CrossEntropy(model, name='ce')

    para_dic = dict(zip(['hidden layer', 'learning rate', 'decay'], [hidden_layer, lr, decay]))

    return model, loss_fun, optimizer, para_dic


def train(model, images, labels, loss, optimizer, num_epochs, para_dic: dict, batch_size=64, train_num=50000, iter_valid=10 ):

    cur_datetime = datetime.now()
    date_str = '{:0>4d}{:0>2d}{:0>2d}'.format(cur_datetime.year, cur_datetime.month, cur_datetime.day)
    time_str = cur_datetime.strftime('%H%M%S')

    model_save_path = './model/' + date_str + '_' + time_str

    if os.path.exists(model_save_path) == False:
        os.makedirs(model_save_path)
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    valid_times = 0    
    max_valid_acc = 0
    best_model_index = None
    best_model_result = dict()

    # iterations per epoch
    batch_num = train_num // batch_size + 1

    train_images = images[:train_num]
    train_labels = labels[:train_num]

    valid_images = images[train_num:]
    valid_labels = labels[train_num:]

    for epoch in tqdm(range(num_epochs)):
        total_acc = []
        total_loss = []

        # training

        for k in range(batch_num):
            if k< batch_num-1:
                X = train_images[k*batch_size: (k+1)*batch_size]
                y = train_labels[k*batch_size: (k+1)*batch_size]
            else:
                X = train_images[k*batch_size: ]
                y = train_labels[k*batch_size: ]
            
            output = model(X)

            L = loss(output, y)
            y_pred = output.argmax(axis=1)

            total_loss.append( L.item() )
            total_acc.append( (y_pred == y).mean() )

            L.backward()
            optimizer.step()
            # optimizer.zero_grad()

            # validate and save Model
            if k % iter_valid == 0:

                valid_times += 1

                valid_output = model(valid_images)
                valid_pred = valid_output.argmax(axis=1)
                valid_loss = loss(valid_output, valid_labels)

                valid_loss_item = valid_loss.item()
                valid_acc = (valid_pred == valid_labels).mean()

                val_loss.append(valid_loss_item)
                val_acc.append(valid_acc)

                train_loss.append(total_loss[k])
                train_acc.append(total_acc[k])

                print("Epoch: {:0>2d}/{:0>2d} Itr: {:0>4d}/{:0>4d} || training loss: {:.8f} , training accuracy :{:.4f} ".format(epoch+1, num_epochs, k+1, batch_num, total_loss[k], total_acc[k]) + 
                  " validation loss: {:.8f} , validation accuracy :{:.4f} ".format(val_loss[valid_times-1], val_acc[valid_times-1]))

                # save best model
                if valid_acc > max_valid_acc:
                    best_model_index = (epoch, k)
                    max_valid_acc = valid_acc
                    best_model_result['training loss'] = total_loss[k]
                    best_model_result['training accuracy'] = total_acc[k]
                    best_model_result['validation loss'] = val_loss[valid_times-1]
                    best_model_result['validation accuracy'] = val_acc[valid_times-1]

                    with open(model_save_path + '/best_model.pkl', 'wb') as f_best:
                        pickle.dump(model, f_best)
        
        with open(model_save_path + '/model_epoch_{:0>2d}.pkl'.format(epoch+1), 'wb') as f:
            pickle.dump(model,f)
        

        

    best_model_str = 'Best Model at (epoch: {:0>2d}, iteration: {:0>4d})'.format(best_model_index[0]+1,best_model_index[1]+1)

    with open(model_save_path + '/log.txt', 'w') as txt_file:
        for key in para_dic.keys():
            para_str = str(key) + ': ' + str(para_dic[key])
            txt_file.writelines(para_str + '\n')
        
        txt_file.writelines('batch size: {:d}'.format(batch_size)+ '\n')
        txt_file.writelines(best_model_str+ '\n')

        for key in best_model_result.keys():
            result_str = str(key) + ': ' + str(best_model_result[key])
            txt_file.writelines(result_str + '\n')
        

    print('Training Complete!')
    print(best_model_str)

    return train_loss, train_acc, val_loss, val_acc