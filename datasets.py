import torch
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset,DataLoader,random_split
from torchvision import datasets,transforms


def getDataset(dataset,batchsize):
    
    match dataset:
        case "mnist":
            train = datasets.MNIST(root='./data',train=True, download=True, transform=transforms.PILToTensor())
            test = datasets.MNIST(root='./data',train=False, download=True, transform=transforms.PILToTensor())

            train_size = int(len(train)*0.85)
            validation_size = int(len(train)*0.15)
            train, validation = random_split(train,[train_size,validation_size])

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader, dataset
        case "fashion-mnist":
            train = datasets.FashionMNIST(root='./data',train=True, download=True, transform=transforms.PILToTensor())
            test = datasets.FashionMNIST(root='./data',train=False, download=True, transform=transforms.PILToTensor())

            train_size = int(len(train)*0.85)
            validation_size = int(len(train)*0.15)
            train, validation = random_split(train,[train_size,validation_size])

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader,dataset
        case "iris":
            iris = load_iris()

            x = torch.tensor(iris.data, dtype = torch.float32)
            y = torch.tensor(iris.target, dtype = torch.long)

            full_iris = TensorDataset(x,y)

            total_size = len(full_iris)
            train_size = int(total_size*0.7)
            validation_size = int(total_size*0.15)
            test_size = total_size - train_size - validation_size

            train,validation,test = random_split(full_iris,[train_size,validation_size,test_size])

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader,dataset