import torch
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset,DataLoader,random_split
from torchvision import datasets,transforms
import copy


def select_dataset(config): #config[data_parameters]
    dataset = config['dataset']
    batchsize = config['batch_size']
    augmentation = config.get('augmentation',False)
        
    match dataset:
        case "MNIST":
            train = datasets.MNIST(root='./data',train=True, download=True, transform=transforms.ToTensor())
            test = datasets.MNIST(root='./data',train=False, download=True, transform=transforms.ToTensor())

            train_data = train.data.float() / 255.0
            mean, std = train_data.mean().item(), train_data.std().item() #표준화를 위해 평균과 표준편차 계산
            train.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([mean],[std])])
            test.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([mean],[std])])
            #compose는 nn.sequential처럼 데이터 전처리 과정에서 원하는 transform 설정을 묶어주는 역할.
            #normalize 시에 mean과 std를 채널에 대해 받으며, rgb 세 개의 채널에 대해 각각 평균과 표준편차를
            #리스트 혹은 튜플형으로 세 개 씩 받는 것을 전제로 하기 때문에 mean과 std를 묶어주는것이 안전함.
            train_size = int(len(train)*0.85)
            validation_size = int(len(train)-train_size)
            train, validation = random_split(train,[train_size,validation_size])

            if augmentation:
                train = copy.deepcopy(train) # 원본 데이터만 복사 = copy.copy, 내부 데이터까지 복사 = copy.deepcopy
                #deepcopy 시에 내부 데이터까지 동일한 완전히 새로운 객체가 생성되므로, 이때 transform 속성을 바꾸면
                #validation 데이터까지 영향을 미치는 것을 막을 수 있다.
                train.dataset.transform = transforms.Compose([
                    transforms.RandomCrop(28, padding=2), #padding을 4로 해서 36*36크기로 키운 다음,
                    #키운 이미지를 랜덤하게 28*28크기로 크롭함. 이과정에서 이미지가 한쪽으로 쏠리는 듯이 변형됨.
                    transforms.RandomRotation(degrees=10),#-15~15도 랜덤하게 기울어짐
                    transforms.ToTensor(),
                    transforms.Normalize([mean],[std])
                ]) 

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader, dataset
        case "fashion-MNIST":
            train = datasets.FashionMNIST(root='./data',train=True, download=True, transform=transforms.ToTensor())
            test = datasets.FashionMNIST(root='./data',train=False, download=True, transform=transforms.ToTensor())

            train_data = train.data.float() / 255.0
            mean, std = train_data.mean().item(), train_data.std().item()
            train.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([mean],[std])])
            test.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([mean],[std])])

            train_size = int(len(train)*0.85)
            validation_size = int(len(train)-train_size)
            train, validation = random_split(train,[train_size,validation_size])

            if augmentation: #데이터 augmentation 할때 사용됨.
                train = copy.deepcopy(train)
                train.dataset.transform = transforms.Compose([
                    transforms.RandomCrop(28, padding=2),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize([mean],[std])
                ])

            train_data_loader = DataLoader(dataset=train, batch_size=batchsize,shuffle=True)
            validation_data_loader = DataLoader(dataset=validation, batch_size=batchsize,shuffle=True)
            test_data_loader = DataLoader(dataset=test, batch_size=batchsize,shuffle=True)

            return train_data_loader,validation_data_loader,test_data_loader,dataset
        case "iris":
            iris = load_iris()

            x = torch.tensor(iris.data, dtype = torch.float32)
            y = torch.tensor(iris.target, dtype = torch.long)

            mean = x.mean(dim=0) #평균
            std = x.std(dim=0) #표준편차
            x = (x-mean)/std #(원본 - 평균) / 표준편차 로 표준화
            #mnist(fashion-mnist)의 경우 데이터가 0~255의 한정된 값을 가지고 있으며, 데이터가 0일 경우가 다수 존재함.
            #0을 표준화 시에 음의 실수가 될 가능성이 있기때문에 정규화를 통해 0을 0으로 유지 가능.
            #또한 0을 w+b에 통과 시켰을 때 b만 남기때문에 relu함수를 통과 시켰을 시 0과 아주 가까운 값(b)이 나오고
            #기울기 또한 0에 가깝기 때문에 0이 자주 나오는 픽셀들에 연결된 w와 b는 사실상 업데이트가 안되는 수준임
            #따라서 학습의 효율 측면에서 정규화가 표준화보다 더 우위에 있을 가능성이 있음.

            #다만, 정규화 시에는 값이 전부 양수이므로, 모든 w를 증가시키거나 감소시키는 방향밖에 선택지가 없음
            #따라서 업데이트 시 minimum으로 향하는 길이 지그재그 형태가 되어버리는 문제가 발생하는데,
            #이는 표준화를 통해 해결할 수 있음. input을 양수와 음수가 적절히 섞여있는 형태로 표준화 하게되면
            #어떤 weight는 증가, 어떤 weight는 감소 식으로 움직이며 minimum으로 향하는 적절한 방향을 잘 찾아가
            #minimum에 도달하는 시간을 줄일 수 있음. 또한 dead ReLU문제도 input을 양과 음 적절하게 가져가며 해결가능
            
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
        

