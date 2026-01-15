import torch
from torch import nn,optim


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# optimizer = optim.Adam(model.parameters(),lr=1e-3)

def model_train(dataloader,model,loss_func,optimizer,device,loss_type = "CE"):
    model.train()

    train_loss_sum = train_correct = train_total = 0
    total_train_batch = len(dataloader)

    for image,label in dataloader:
        x_train = image.to(device)
        y_train = label.to(device)

        output = model(x_train)
        if loss_type == "MSE":
            output = nn.functional.softmax(output,dim=1) #다시 공부
            y_one_hot = nn.functional.one_hot(y_train,num_classes=output.size(1)).float() #다시공부
            loss_func = nn.MSELoss()
            loss = loss_func(output,y_one_hot)
        else:
            loss = loss_func(output,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() #전체 loss

        train_total += y_train.size(0) # 트레이닝 횟수 저장. batchsize를 계속해서 더해줌 (마지막에는 나머지가 들어가므로 total = 51000)
        train_correct += ((torch.argmax(output,1)) == y_train).sum().item() #output이 정답과 일치하는 횟수를 전부 저장

    
    train_avg_loss = train_loss_sum / total_train_batch # 트레이닝 횟수 1500회 언저리 (배치사이즈 말고)
    train_avg_accuracy = 100*train_correct / train_total # 51000회중 맞힌 횟수에 100을 곱해 %로 변환해줌

    return train_avg_loss,train_avg_accuracy