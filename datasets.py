import torch
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset,DataLoader,random_split
from torchvision import datasets,transforms


def getDataset(config): #config[data_parameters]
    dataset = config['dataset']
    batchsize = config['batchsize']
    match dataset:
        case "MNIST":
            train = datasets.MNIST(root='./data',train=True, download=True, transform=transforms.ToTensor())
            test = datasets.MNIST(root='./data',train=False, download=True, transform=transforms.ToTensor())

            train_size = int(len(train)*0.85)
            validation_size = int(len(train)-train_size)
            train, validation = random_split(train,[train_size,validation_size])

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader, dataset
        case "fashion-MNIST":
            train = datasets.FashionMNIST(root='./data',train=True, download=True, transform=transforms.ToTensor())
            test = datasets.FashionMNIST(root='./data',train=False, download=True, transform=transforms.ToTensor())

            train_size = int(len(train)*0.85)
            validation_size = int(len(train)-train_size)
            train, validation = random_split(train,[train_size,validation_size])

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader,dataset
        case "iris":
            iris = load_iris()

            x = torch.tensor(iris.data, dtype = torch.float32)
            y = torch.tensor(iris.target, dtype = torch.long)

            # y = torch.nn.functional.one_hot(y, num_classes=3).float()


            mean = x.mean(dim=0) #평균
            std = x.std(dim=0) #표준편차
            x = (x-mean)/std #(원본 - 평균) / 표준편차 로 표준화
            #mnist(fashion-mnist)의 경우 데이터가 0~255의 한정된 값을 가지고 있으며, 데이터가 0일 경우가 다수 존재함.
            #0을 표준화 시에 음의 실수가 될 가능성이 있기때문에 정규화를 통해 0을 0으로 유지 가능.
            #또한 0을 w+b에 통과 시켰을 때 b만 남기때문에 relu함수를 통과 시켰을 시 0과 아주 가까운 값(b)이 나오고
            #기울기 또한 0에 가깝기 때문에 0이 자주 나오는 픽셀들에 연결된 w와 b는 사실상 업데이트가 안되는 수준임
            #따라서 학습의 효율 측면에서 데이터가 0인 픽셀을 굳이 음수의 실수로 만들어서 (정규화) 바깥쪽 픽셀도
            #음수의 값에 따른 업데이트를 진행한다면 minimum에 도달하는 속도가 느려질 것임. 이는 신경쓸 필요가 적은
            #픽셀에 대해서도 업데이트를 계속 진행하기때문에 minimum까지 도달하는 데에 있어서 올바른 방향이 아닌
            #지그재그 형태로 움직이며 업데이트하게 되고, 학습 효율이 떨어짐.
            
            #iris를 정규화가 아닌 표준화 하는 이유는 iris의 경우 데이터가 한정된 값을 갖지 않으며, 만약 
            #평균적인 값보다 훨씬 큰 이상점이 데이터에 존재했을 때 정규화 시에 대부분의 값이 0 근처로 몰리는
            #현상을 피할 수 있도록 표준화 하는것을 지향함.

            full_iris = TensorDataset(x,y)

            total_size = len(full_iris)
            train_size = int(total_size*0.7)
            validation_size = int(total_size*0.15)
            test_size = total_size - train_size - validation_size #test는 나머지 15%

            train,validation,test = random_split(full_iris,[train_size,validation_size,test_size])

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader,dataset
        

# train_data_loader, validation_data_loader, test_data_loader, dataset = data_select()