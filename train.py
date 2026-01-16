import torch
from torch import nn,optim


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def select_optimizer(model,config): #config[train_parameters]
    learning_rate = config['lr']
    optimizer_type = config.get('optim','Adam')
    if optimizer_type == "SGD":
        return optim.SGD(model.parameters(),lr=learning_rate)
    else:
        return optim.Adam(model.parameters(),lr=learning_rate)


def model_train(dataloader,model,optimizer,device,config): #config[train_parameters]

    model.train()#dropout과 bn에 대해서 전부 활성화가 되도록 레이어 동작 방식을 설정함

    loss_type = config.get('loss_type','CE')

    train_loss_sum = train_correct = train_total = 0
    total_train_batch = len(dataloader)

    if loss_type == "MSE":
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    for image,label in dataloader:
        x_train = image.to(device)
        y_train = label.to(device)

        output = model(x_train)
        if loss_type == "MSE":
            output = nn.functional.softmax(output,dim=1) #지정한 dimension의 값을 다 더했을 때 1이 되도록 softmax
            #여기서는 output인 tensor[batchsize,3]중 3을 타겟해 softmax해야 하므로 dim = 1 로 지정
            #마지막 차원을 dim = -1 로 지정할 수도 있음
            y_one_hot = nn.functional.one_hot(y_train,num_classes=output.size(1)).float()
            #one_hot은 숫자로 된 인덱스를 0과 1로 이루어진 벡터형으로 만들어줌
            #long타입의 tensor , 전체 클래스 개수를 인자로 넣어줘야 하며 num_classes = output의 1번차원 size를 지정
            #.float()를 활용해 one_hot으로 생성한 벡터를 float형으로 바꿔줌. loss를 계산할 때 y-yhat을 수행해야 하기
            #때문에 output과 동일한 float형으로 변환함.
            loss = loss_func(output,y_one_hot)
        else:
            loss = loss_func(output,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() #전체 loss

        train_total += y_train.size(0) # 트레이닝 횟수 저장. batchsize를 계속해서 더해줌
        train_correct += ((torch.argmax(output,1)) == y_train).sum().item() #output이 정답과 일치하는 횟수를 전부 저장

    
    train_avg_loss = train_loss_sum / total_train_batch # 트레이닝 횟수 (전체데이터 / 배치사이즈)
    train_avg_accuracy = 100*train_correct / train_total # 51000회중 맞힌 횟수에 100을 곱해 %로 변환

    return train_avg_loss,train_avg_accuracy

def model_evaluate(dataloader,model,device,config):
    model.eval() #dropout은 비활성화 하여 모든 뉴런을 사용하며, bn은 학습 시에 저장해둔
    #각 평균과 분산을 가져와서 사용함. test,validation데이터에 대해 통계를 쓰지 않으며,
    #학습 시에 진행해둔 

    loss_type = config.get('loss_type','CE')

    with torch.no_grad(): #autograd에게 미분계산할 필요 없다고 알려줌. 각 tensor의 grad속성이 업데이트되지 않음.

        val_loss_sum = val_correct = val_total = 0
        total_val_batch = len(dataloader)
        if loss_type == "MSE":
            loss_func = nn.MSELoss()
        else:
            loss_func = nn.CrossEntropyLoss()
        
        for image,label in dataloader: 
            x_val = image.to(device)
            y_val = label.to(device)

            output = model(x_val)
            if loss_type == "MSE":
                output = nn.functional.softmax(output,dim=1) 
                y_one_hot = nn.functional.one_hot(y_val,num_classes=output.size(1)).float()
                loss = loss_func(output,y_one_hot)
            else:
                loss = loss_func(output,y_val)
            val_loss_sum += loss.item()

            val_total += y_val.size(0)
            val_correct += ((torch.argmax(output,1)) == y_val).sum().item() 
        
    val_avg_loss = val_loss_sum / total_val_batch
    val_avg_accuracy = 100*val_correct / val_total 

    return val_avg_loss,val_avg_accuracy

class EarlyStopping(): #함수가 아닌 class로 정의하는 이유는 이전 state를 기억하기 위함 (best_score, epoch)
    def __init__(self,patience = 5, delta = 0.0, path = 'checkpoint.pt',verbose = True):
        self.patience = patience #성능이 향상되지 않아도 참을 횟수
        self.delta = delta #성능 향상을 인정할 최소 값
        self.path = path #모델(체크포인트) 저장 경로
        self.verbose = verbose #관련 메시지 출력 여부

        self.count = 0 #patience 카운트
        self.best_score = None #역대 최고 점수(최저 loss)
        self.early_stop = False #early stop 신호 (True 시 종료)
        self.val_loss_min = float('inf') #비교를 위한 최저치 loss값 (초기 최저치는 무한으로 설정)

    def save_checkpoint(self, val_loss, model): #체크포인트 저장 함수
        if self.verbose :
            print(f"Validation loss decreased to {val_loss}. save model\n")
            torch.save(model.state_dict(),self.path) #torch.save는 객체를 경로로 저장할 수 있음. 리스트, 딕셔너리, tensor모두 저장 가능
            #state_dict는 특정 층의 어떤 변수가 어떤 값을 갖고 있는지를 전부 저장함. 
            #sequential을 사용해 init했을 시 sequential이름.순서(0..),파라미터(weight, bais 등)으로 생성되며,
            #fc,relu등 변수를 하나하나 설정해 init했다면 변수이름(fc1).weight, fc1.bias 처럼 생성됨
            #torch.load('저장파일.pt', map_loaction='모델을 올릴 위치')를 통해 모델을 불러올 수 있음
            #만약 sequential로 저장한 모델을 불러 올 시에, stack.0.weight -> fc1.0.weight처럼 이름이 바뀌어 있다면 불러올 수 없음
            #또한 sequential은 특정 층을 통과중인 값을 뽑아서 다른곳에 쓰거나 입력값을 나중에 더해주는 등의 구현이 어려움.
            #따라서 단순한 구조라면 sequential을 활용해서 구현하는 것이 가독성 측면에서 우위지만, 복잡한 구조에서는 forward를 하나하나 짜는게 좋음
            self.val_loss_min = val_loss

    def __call__(self, val_loss, model):
        score = -val_loss #loss가 낮을수록 좋으므로 음수로 바꿔서 점수로 환산
        
        if self.best_score is None: #초기 설정
            self.best_score = score
            self.save_checkpoint(val_loss,model)
        elif score < self.best_score + self.delta: #성능향상 x
            self.count += 1
            if self.verbose:
                print(f"EarlyStopping count: {self.count}\n")
            if self.count >= self.patience:
                self.early_stop = True
        else: #성능향상 o
            self.best_score = score
            self.save_checkpoint(val_loss,model)
            self.count = 0