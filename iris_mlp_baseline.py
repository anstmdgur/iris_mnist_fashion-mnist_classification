import torch 
import datasets
import model
import train
import eval
import yaml

import random
import os
import numpy as np

import time

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def main(config_name):
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./as_lab_project_1/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = config[config_name]
    data_parameters = config['data_parameters']
    model_parameters = config['model_parameters']
    train_parameters = config['train_parameters']

    train_data_loader,validation_data_loader,test_data_loader,dataset = datasets.select_dataset(data_parameters)
    my_model = model.select_model(model_parameters,device)
    optimizer = train.select_optimizer(my_model,train_parameters)
    scheduler = train.select_scheduler(optimizer,train_parameters)
    early_stop = train.EarlyStopping() #iris patience
    EPOCH = train_parameters['epochs']

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_time = time.time()
    real_epoch = 0

    print(f"{config_name} training start.")

    for ep in range(1,EPOCH+1):
        real_epoch = ep

        train_avg_loss, train_avg_accuracy = train.model_train(train_data_loader,my_model,optimizer,scheduler,device,train_parameters)
        val_avg_loss, val_avg_accuracy = train.model_evaluate(validation_data_loader,my_model,device,train_parameters)

        history['train_loss'].append(train_avg_loss)
        history['val_loss'].append(val_avg_loss)
        history['train_acc'].append(train_avg_accuracy)
        history['val_acc'].append(val_avg_accuracy)
        if ep % 5 == 0:
            print(f"epoch = {ep} train loss = {train_avg_loss} train acc = {train_avg_accuracy} val loss = {val_avg_loss} val acc = {val_avg_accuracy}\n")

        early_stop(val_avg_loss,my_model,config_name,dataset)
        if early_stop.early_stop:
            print("Early stop. Training Finish.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_epoch = total_time/real_epoch
    print(f"total time = {total_time}sec.   avg time per epoch = {avg_time_per_epoch}sec.\n")

    match dataset:
        case 'iris':
            classes = ['Setosa', 'Versicolor', 'Virginica']
        case 'MNIST':
            classes = [str(i) for i in range(10)]
        case 'fashion-MNIST':
            classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

    my_model.load_state_dict(torch.load(f"./as_lab_project_1/checkpoint/{dataset}/{config_name}_checkpoint.pt"))
    train_loss, train_acc = train.model_evaluate(train_data_loader,my_model,device,train_parameters)
    val_loss, val_acc = train.model_evaluate(validation_data_loader,my_model,device,train_parameters)
    test_loss, test_acc = train.model_evaluate(test_data_loader,my_model,device,train_parameters)
    
    with open(f"./as_lab_project_1/log/{dataset}/{config_name}_result.txt",'w',encoding='utf-8') as f:
        f.write(f"{config_name} model\n\n")
        f.write(f"Train : Acc {train_acc:.2f}% | Loss {train_loss:.4f}\n")
        f.write(f"Val   : Acc {val_acc:.2f}% | Loss {val_loss:.4f}\n")
        f.write(f"Test  : Acc {test_acc:.2f}% | Loss {test_loss:.4f}\n\n")
        f.write(f"total time = {total_time}sec.   avg time per epoch = {avg_time_per_epoch}sec.\n\n")

    eval.plot_history(history,config_name,dataset)
    eval.plot_confusion_matrix_and_report(my_model,test_data_loader,device,classes,config_name,dataset)
    print(f"{config_name} training end.\n")


main("iris_mlp_baseline")
main("iris_mlp_adam")
main("iris_mlp_low_batch")
main("iris_mlp_high_batch")
main("iris_mlp_low_lr")
main("iris_mlp_high_lr")
main("iris_mlp_no_standardization")
