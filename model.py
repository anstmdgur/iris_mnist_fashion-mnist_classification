
from torch import nn

#lr, batchsize, optimizer, lr scheduler, dropout p, weight regularization?, relu

class IrisMlpModel(nn.Module): #4->16->3 bn,dropout 선택적용
    def __init__(self,dropout_p,bn=False):
        super().__init__()
        layer_list = [nn.Linear(4,16)]
        if bn:
            layer_list.append(nn.BatchNorm1d(16))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Dropout(dropout_p))
        layer_list.append(nn.Linear(16,3))
        self.stack = nn.Sequential(*layer_list)
    def forward(self,x):
        prediction = self.stack(x)
        return prediction
    

class MnistMlpModel(nn.Module): #784->256->10 bn,dropout 선택적용
    def __init__(self,dropout_p,bn=False):
        super().__init__()
        layer_list = [nn.Flatten(),nn.Linear(784,256)]
        if bn:
            layer_list.append(nn.BatchNorm1d(256))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Dropout(dropout_p))
        layer_list.append(nn.Linear(256,10))
        self.stack = nn.Sequential(*layer_list)
    def forward(self,x):
        prediction = self.stack(x)
        return prediction
    
class CnnModel(nn.Module): 
    # feature map 개수 : 1(이미지 input)->32->64
    # feature map 크기 : 28*28(이미지 원본 크기) -> 14*14 -> 7*7 (padding적용으로 conv 시 이미지 크기 손실 x)
    def __init__(self,dropout_p,bn=False):
        super().__init__()
        layer_list = [nn.Conv2d(in_channels= 1, out_channels= 32, kernel_size=3, padding= 1)]
        if bn:
            layer_list.append(nn.BatchNorm2d(32))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
        layer_list.append(nn.Dropout(dropout_p))
        
        layer_list.append(nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3, padding= 1))
        if bn:
            layer_list.append(nn.BatchNorm2d(64))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
        layer_list.append(nn.Dropout(dropout_p))

        layer_list.append(nn.Flatten())
        layer_list.append(nn.Linear(64*7*7,256))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Dropout(dropout_p))
        layer_list.append(nn.Linear(256,10))
        self.stack = nn.Sequential(*layer_list)
    
    def forward(self,x):
        prediction = self.stack(x)
        return prediction
    
def select_model(config,device): #config [model_parameters]
    parameters = config.copy()
    model_type = parameters.pop('model')

    if model_type == "iris_MLP":
        return IrisMlpModel(**parameters).to(device)
    elif model_type == "MNIST_MLP":
        return MnistMlpModel(**parameters).to(device)
    elif model_type == "CNN":
        return CnnModel(**parameters).to(device)

