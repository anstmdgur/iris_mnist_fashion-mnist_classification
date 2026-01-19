import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns



def plot_history(history, save_path = './as_lab_project_1/plot/training_result.png'): # 학습과정의 그래프 저장
    #history변수는 메인 모듈에서 train/val의 loss/acc를 각각 저장함 총 4개의 key 존재
    epochs = range(1,(len(history['train_loss']) + 1)) #모델 학습 중 실시한 epoch을 가져옴 (x축)
    plt.figure()
    plt.subplot(1,2,1) #1행 2열로 나누고, 그 중 1번칸 사용
    plt.plot(epochs, history['train_loss'],'b-',label = 'Training Loss')
    plt.plot(epochs, history['val_loss'],'r-',label = 'Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label = 'Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label = 'Validation Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

def plot_confusion_matrix(model,dataloader,device,classes,save_path = './as_lab_project_1/plot/confusion_matrix.png'):
    #모델이 어떤 데이터를 어떻게 틀렸는지 알기 위해서 사용함. 이때 최종모델에 test를 돌리는 것처럼
    #confusion matrix를 위한 예측과 정답 데이터를 model을 통해 돌리면서 알아봄.
    #classes변수는 0,1,2 같은 인덱스 번호가 아닌 이름으로 표를 구성할 수 있도록 해줌. 0->'setosa' 처럼.
    #메인 모듈에서 classes 변수를 리스트로 만든 뒤, 전달해주면 됨.
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            prediction = torch.argmax(output,dim=1) #예측중 가장 큰 값 (softmax통과하지 않더라도 가장 큰 값이 모델의 예측)의 index를 가져옴
            #dimension은 1로, -1로 지정해도 무방함. output이 [batchsize, 정답 개수] 형태이기 때문에
            all_predictions.extend(prediction.cpu().numpy()) #confusion_matrix함수는 tensor가 아닌 numpy형태의 데이터가 필요함
            all_labels.extend(label.cpu().numpy()) #추가로 cpu로 가져오지 못하면 numpy로 변환이 불가능하기 때문에 cpu로 이동시킴
    cm = confusion_matrix(all_labels,all_predictions) #extend로 길어진 정답리스트,예측리스트를 인자로 받음
    # 같은 인덱스끼리 비교하면서 정답:9, 예측 7이라면 9행 7열의 숫자를 +1함. 결국 10*10 크기의 행렬로 만들어짐

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path)

def plot_misclassified_images(model,dataloader,device,classes,num_images = 25,save_path = './as_lab_project_1/plot/misclassified_images.png'):
    model.eval()
    #data augmentation을 통해서 기울어지거나 노이즈가 있는 이미지에 대해서도 학습시켜 모델의 견고성을 높일수있다.
    misclassified_images = []
    misclassified_true = []
    misclassified_predict = []

    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            predict = torch.argmax(output, dim=1)
            index = (predict != label).nonzero().squeeze()
            #nonzero는 zero가 아닌 (False가 아닌. 여기서는 틀린 예측) 곳의 [틀린개수,차원 수] 형태 tensor반환
            #3,5 인덱스가 틀렸다면 2행(2개 틀림) 1열(지금 다루는데이터가 1차원 데이터라서)의 tensor를 반환
            #squeeze는 값이 1인 차원을 전부 제거함. 따라서 index에는 [3,5]처럼 틀린 인덱스만 남게됨.
            #추가로 unsqueez는 인자로 주는 차원에 값이 1인 차원을 추가해줌 unsqueeze(-1) = [16,7,7] -> [16,7,7,1]
            if index.ndim == 0: #batchsize중 하나만 틀렸을 경우(스칼라일 경우. 0차원일 경우)
                index = index.view(1) #view를 통해 1차원 tensor로 변환함.
            
            for idx in index:
                if len(misclassified_images) < num_images:
                    misclassified_images.append(image[idx].cpu()) 
                    misclassified_true.append(label[idx].cpu().item())
                    misclassified_predict.append(predict[idx].cpu().item())
                else:
                    break
            if len(misclassified_images) >= num_images:
                break
        if not misclassified_images:
            print("틀린 이미지가 없습니다.")
            return
            
        rows = cols = 5
        plt.figure(figsize=(10, 10))
        for i in range(len(misclassified_images)):
            plt.subplot(rows, cols, i + 1)
            img_tensor = misclassified_images[i].squeeze(0) #채*행*열 -> 행*열 로 변환
            plt.imshow(img_tensor, cmap='gray')
            true_name = classes[misclassified_true[i]]
            predict_name = classes[misclassified_predict[i]]
            plt.title(f"T: {true_name}\nP: {predict_name}")
        plt.tight_layout()
        plt.savefig(save_path)