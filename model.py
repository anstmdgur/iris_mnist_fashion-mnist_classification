import torch
from torch import nn
import datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_select():
    datalist = ["iris","mnist","fashion-mnist"]
    while True:
        dataset = input("사용할 데이터셋 입력: (iris, mnist, fashion-mnist)\n").lower()
        if dataset not in datalist:
            print("입력오류. iris, mnist, fashion-mnist 중 선택해 입력하세요.\n")
            continue
        try:
            batchsize = int(input("batch size 입력: "))
            if batchsize < 1:
                print("입력오류. batch size는 0보다 큰 정수여야 합니다.\n")
                continue
            return datasets.getDataset(dataset, batchsize)
        except ValueError:
            print("입력오류. batch size는 0보다 큰 정수여야 합니다.\n")

train_data_loader, validation_data_loader, test_data_loader, dataset = data_select()

#lr, batchsize, optimizer, lr scheduler, dropout p, weight regularization, hidden layer 깊이 및 노드 수, 활성함수

class MlpModel(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_layer,dropout_p,bn=False):
        super().__init__()
        layer_list = [nn.Flatten()]
        prev = input_dim
        for hidden_dimension in hidden_layer:
            layer_list.append(nn.Linear(prev,hidden_dimension))
            if bn:
                layer_list.append(nn.BatchNorm1d(hidden_dimension))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(dropout_p))
            prev = hidden_dimension
        layer_list.append(nn.Linear(prev,output_dim))
        self.stack = nn.Sequential(*layer_list)
    def forward(self,x):
        prediction = self.stack(x)
        return prediction
    
class CnnModel(nn.Module):
    def __init__(self,flatten_size,output_dim,fc_hidden_dim,hidden_layer,dropout_p,dropout_output_p,bn=False):
        super().__init__()
        layer_list = []
        prev = 1
        for hidden_dimension in hidden_layer:
            layer_list.append(nn.Conv2d(in_channels= prev, out_channels= hidden_dimension, kernel_size=3, padding= 1))
            if bn:
                layer_list.append(nn.BatchNorm2d(hidden_dimension))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
            layer_list.append(nn.Dropout(dropout_p))
            prev = hidden_dimension
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Linear(flatten_size,fc_hidden_dim))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Dropout(dropout_output_p))
        layer_list.append(nn.Linear(fc_hidden_dim,output_dim))
        self.stack = nn.Sequential(*layer_list)
    
    def forward(self,x): #이미지는 1*28*28
        prediction = self.stack(x)
        return prediction
    
def model_select(config):
    model_list = ["CNN","MLP"]
    while True:
        model = input("사용할 모델 입력: (CNN, MLP)\n").upper()
        if model not in model_list:
            print("입력오류. CNN,MLP 중 선택해 입력하세요.\n")
            continue
        elif dataset == "iris" and model == "CNN":
            print("iris dataset은 CNN모델로 학습할 수 없습니다.\n")
            continue
        break
    match model:
        case "CNN":
            model = CnnModel(**config).to(device)
            return model
        case "MLP":
            model = MlpModel(**config).to(device)
            return model

