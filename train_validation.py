import sys, os
import torch
import torch.nn as nn
from models.dl_bbbp import DL_BBBP
from models.dl_bbbp_onlygraph import DL_BBBP_onlygraph
from models.dl_bbbp_onlymaccs import DL_BBBP_onlymaccs
from utils import *
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))
    #torch.cuda.empty_cache () 

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).detach().cpu()), 0)
            print('Test Loss: {:.6f}'.format(loss.item()))
    #torch.cuda.empty_cache () 
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


modeling = DL_BBBP  # DL_BBBP_onlygraph    DL_BBBP_onlymaccs
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 8
NUM_EPOCHS = 50
print('Epochs: ', NUM_EPOCHS)

processed_data_file = 'data/processed/dataset.pt'
if not os.path.isfile(processed_data_file):
    print('please run create_data.py to prepare data first.')
else:
    # 加载文件
    dataset_data = TestbedDataset(root='data', dataset='dataset')
    dataset_length = len(dataset_data)      # 总长度
    k = 10                                  # 设定10折
    valid_length = int(len(dataset_data)/k) # valid集的长度

    for fold in range(k):
        print('Running in ', fold, 'fold.')
        # 创建索引列表
        valid_idx = list(range(fold*valid_length, fold*valid_length+valid_length))
        train_idx = list(range(dataset_length))
        del train_idx[fold*valid_length : fold*valid_length+valid_length]
        # 划分数据集
        valid_data = torch.utils.data.Subset(dataset_data, valid_idx)
        valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)
        train_data = torch.utils.data.Subset(dataset_data, train_idx)
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        # 实例化模型，设定参数
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        best_acc = 0
        best_epoch = -1
        model_file_name = './result/model_' + str(fold) + '_fold.model'
        result_file_name = './result/result_' + str(fold) + '_fold.csv'
        ACC_pic_name = './result/ACC_' + str(fold) + '_fold.png'
        AUC_pic_name = './result/AUC_' + str(fold) + '_fold.png'
        x = []  # epoch
        y = []  # ACC
        z = []  # AUC
        # 训练
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            labels,predicts = predicting(model, device, valid_loader)
            ret = [acc(labels,predicts), auc(labels,predicts)]
            x.append(epoch)
            y.append(ret[0])
            z.append(ret[1])
            if ret[0] > best_acc:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret)))
                best_epoch = epoch+1
                best_acc = ret[0]
                print('acc improved at epoch ', best_epoch,
                      '; best_acc:',best_acc, '; auc:',ret[1], model_st)
            else:
                print('acc:',ret[0], 'No improvement since epoch ', best_epoch,
                      '; best_acc:',best_acc, '; auc:',ret[1], model_st)
        plt.figure(figsize=(10,6))
        plt.plot(x, y)
        plt.xlabel("Epochs")
        plt.ylabel("ACC")
        plt.savefig(ACC_pic_name)
        plt.figure(figsize=(10,6))
        plt.plot(x, z)
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.savefig(AUC_pic_name)