import torch
import datasets
import model
import train
import eval
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./as_lab_project_1/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = config['iris_mlp']
data_parameters = config['data_parameters']
model_parameters = config['model_parameters']
train_parameters = config['train_parameters']

train_data_loader,validation_data_loader,test_data_loader,dataset = datasets.select_dataset(data_parameters)
my_model = model.select_model(model_parameters,device)
optimizer = train.select_optimizer(my_model,train_parameters)
early_stop = train.EarlyStopping()
EPOCH = train_parameters['epochs']

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for ep in range(1,EPOCH+1):
    train_avg_loss, train_avg_accuracy = train.model_train(train_data_loader,my_model,optimizer,device,train_parameters)
    val_avg_loss, val_avg_accuracy = train.model_evaluate(validation_data_loader,my_model,device,train_parameters)

    history['train_loss'].append(train_avg_loss)
    history['val_loss'].append(val_avg_loss)
    history['train_acc'].append(train_avg_accuracy)
    history['val_acc'].append(val_avg_accuracy)

    print(f"epoch = {ep} train loss = {train_avg_loss} trian accuracy = {train_avg_accuracy} validation loss = {val_avg_loss} validation accuracy = {val_avg_accuracy}\n")

    early_stop(val_avg_loss,my_model)
    if early_stop.early_stop:
        print("Early stopping triggered. Training Finished.")
        break

match dataset:
    case 'iris':
        classes = ['Setosa', 'Versicolor', 'Virginica']
    case 'MNIST':
        classes = [str(i) for i in range(10)]
    case 'fashion-MNIST':
        classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

eval.plot_history(history)
eval.plot_confusion_matrix(my_model,test_data_loader,device,classes)