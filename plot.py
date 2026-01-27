import torch 
import datasets
import model
import train
import eval
import yaml

import matplotlib.pyplot as plt

import random
import os
import numpy as np

import time

# 그래프를 누적해서 그리기 위해 함수 밖에서 Figure 생성
plt.figure(figsize=(16, 8), dpi=300)

# [추가됨 1] 그래프 순서를 세기 위한 전역 변수
plot_cnt = 0

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def main(config_name, color):
    # [추가됨 2] 전역 변수 사용 선언
    global plot_cnt

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./as_lab_project_1/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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

    print(f"{config_name} training start.")

    for ep in range(1,EPOCH+1):
        real_epoch = ep

        train_avg_loss, train_avg_accuracy = train.model_train(train_data_loader,my_model,optimizer,scheduler,device,train_parameters)
        val_avg_loss, val_avg_accuracy = train.model_evaluate(validation_data_loader,my_model,device,train_parameters)

        history['train_loss'].append(train_avg_loss)
        history['val_loss'].append(val_avg_loss)
        history['train_acc'].append(train_avg_accuracy)
        history['val_acc'].append(val_avg_accuracy)
        if ep % 5 == 0:
            print(f"epoch = {ep} train loss = {train_avg_loss} train acc = {train_avg_accuracy} val loss = {val_avg_loss} val acc = {val_avg_accuracy}\n")

        early_stop(val_avg_loss,my_model,config_name,dataset)
        if early_stop.early_stop:
            print("Early stop. Training Finish.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_epoch = total_time/real_epoch
    print(f"total time = {total_time}sec.    avg time per epoch = {avg_time_per_epoch}sec.\n")

    # --- 그래프 그리기 및 최적 지점 표시 ---
    epochs = range(1, len(history['val_loss']) + 1)
    
    plt.plot(epochs, history['val_loss'], color, label=f'{config_name}')

    best_loss_idx = np.argmin(history['val_loss']) 
    best_epoch = best_loss_idx + 1
    best_val_loss = history['val_loss'][best_loss_idx]

    plt.scatter(best_epoch, best_val_loss, color=color[0], s=50, zorder=5)
    
    # [수정됨 3] plt.text 대신 plt.annotate 사용
    # plot_cnt가 증가할 때마다 텍스트 위치(xytext)를 y축으로 20포인트씩 위로 올립니다.
    # arrowprops: 텍스트가 멀어져도 점과 텍스트를 선으로 이어줍니다.
    plt.annotate(
        f"Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}", # 표시할 텍스트
        xy=(best_epoch, best_val_loss),                    # 점의 위치 (화살표 끝)
        xytext=(10, 20 + plot_cnt * 20),                   # 텍스트 위치 (점보다 얼마나 떨어질지) -> cnt에 따라 높아짐
        textcoords='offset points',                        # 좌표 기준 (포인트 단위 오프셋)
        ha='left', va='bottom',                            # 정렬
        fontsize=9, fontweight='bold',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray') # 화살표 설정
    )

    plt.ticklabel_format(style='plain', axis='x')
    plt.locator_params(axis='x', nbins=20)
    
    # [추가됨 4] 다음 그래프를 위해 카운트 증가
    plot_cnt += 1
    
    return dataset
    
# dataset = main('iris_mlp_adam', 'r-')    
# main('iris_mlp_baseline', 'b-')          
# main('iris_mlp_high_lr', 'g-')             
# main('iris_mlp_low_lr', 'k-')               
# main('iris_mlp_high_batch', 'm-')       
# main('iris_mlp_low_batch', 'c-')       
# main('iris_mlp_no_standardization', 'y-')

dataset = main('mnist_mlp_baseline', 'r-')    
main('mnist_mlp_one_hot', 'b-')          
main('mnist_mlp_no_standardization', 'g-')             
main('mnist_cnn', 'k-')
main('mnist_cnn_depth_2', 'm-')       

plt.legend()
plt.title(f'{dataset} Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.tight_layout() 
plt.savefig(f"./as_lab_project_1/plot/val_loss_plot/{dataset}/{dataset}_val_loss.png",bbox_inches='tight')