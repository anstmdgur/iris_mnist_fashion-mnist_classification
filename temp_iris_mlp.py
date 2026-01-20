import torch
import datasets
import model
import train
import eval
import yaml

import random #시드 고정을 위함
import os
import numpy as np

import time #시간 측정을 위함

#실제 무작위성이 발생하는 시점은 함수를 call했을때 이므로, import후 해당 소스의 함수나 클래스를 사용하기 전에
#seed를 고정해주면 그 이후 호출(call)된 함수나 클래스는 고정된 시드의 영향을 받음.
def set_seed(seed):
    random.seed(seed) #기본 파이썬 내장 random 시드 고정
    os.environ['PYTHONHASHSEED'] = str(seed) #파이썬의 hash함수 동작을 위한 시드 고정
    np.random.seed(seed) # numpy의 무작위성을 위한 seed 고정
    torch.manual_seed(seed) #pytorch의 cpu 텐서 생성과 연산에 대한 시드 고정
    torch.cuda.manual_seed(seed) #pytorch의 gpu(cuda) 텐서 생성과 연산에 대한 시드 고정
    torch.backends.cudnn.deterministic = True # 딥러닝 연산 시 동일한 알고리즘 사용을 강제하여 결과를 같게 유도
    torch.backends.cudnn.benchmark = False # 하드웨어 환경에 맞는 알고리즘을 찾는 벤치마크를 꺼서 결과를 같게 유도

#cuDNN이란 : 딥러닝에서 자주 쓰이는 연산을 최적화 해둔 도구. relu, conv연산 등 함수를 모아둔 라이브러리 처럼 생김
#같은 conv연산이라도, 데이터와 하드웨어어 따라 더 빠른 방법이 다르기때문에 벤치마킹 후 가장 빠른 알고리즘을 선택해
#사용하는 기능도 있음. 이 기능을 꺼줘야 모든 데이터에 대해 같은 알고리즘을 사용해서 재현성이 올라감.
#동일한 input과 output을 갖지만 이 기능을 꺼야 하는 이유는 float 자료형은 32비트를 사용해 아주 정확한 실수를 저장하지
#못하기 때문에, 연산할때 연산 순서가 바뀐다면 값도 아주 미세하게 바뀔 가능성이 있기 때문임. 

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./as_lab_project_1/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config_name = "iris_mlp"
config = config[config_name]
data_parameters = config['data_parameters']
model_parameters = config['model_parameters']
train_parameters = config['train_parameters']

train_data_loader,validation_data_loader,test_data_loader,dataset = datasets.select_dataset(data_parameters)
my_model = model.select_model(model_parameters,device)
optimizer = train.select_optimizer(my_model,train_parameters)
scheduler = train.select_scheduler(optimizer,train_parameters)
early_stop = train.EarlyStopping()
EPOCH = train_parameters['epochs']

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

start_time = time.time()
real_epoch = 0

for ep in range(1,EPOCH+1):
    real_epoch = ep

    train_avg_loss, train_avg_accuracy = train.model_train(train_data_loader,my_model,optimizer,scheduler,device,train_parameters)
    val_avg_loss, val_avg_accuracy = train.model_evaluate(validation_data_loader,my_model,device,train_parameters)

    history['train_loss'].append(train_avg_loss)
    history['val_loss'].append(val_avg_loss)
    history['train_acc'].append(train_avg_accuracy)
    history['val_acc'].append(val_avg_accuracy)

    print(f"epoch = {ep} train loss = {train_avg_loss} train acc = {train_avg_accuracy} val loss = {val_avg_loss} val acc = {val_avg_accuracy}\n")

    early_stop(val_avg_loss,my_model,config_name)
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

eval.plot_history(history,config_name)
eval.plot_confusion_matrix_and_report(my_model,test_data_loader,device,classes,config_name)
eval.plot_misclassified_images(my_model,test_data_loader,device,classes,config_name)