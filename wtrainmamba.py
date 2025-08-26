import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models import mamba
from mutils import train_one_epoch, evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay
import time

#针对mamba中class2容易被判断为class0，调整权重

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Emotion recognition task')
    parser.add_argument(
        '--model',
        help='select network',
        choices=model_dict.keys(),
        default='mamba'
        #mamba
        )
    parser.add_argument('--batch', type=int, help='batch_size', default=64)
    parser.add_argument('--epoch', type=int, help='epoch', default=30)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.01)
    args = parser.parse_args()
    return args


def main(args):
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU可用")
    else:
        device = torch.device("cpu")
        print("GPU不可用，将使用CPU")



#========读取数据=========
    data = pd.read_csv('./WESAD/wesad_ecg_3class.csv', header=None)
    X = data.iloc[:, :-1].values  # 除最后一列外都是特征
    y = data.iloc[:, -1].values  # 最后一列是标签

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将数据转换为PyTorch张量，并移动到GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.int64).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.int64).to(device)

    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(y_train.shape)
    '''
    torch.Size([1322, 1, 700, 1])
    torch.Size([5288, 1, 700, 1])
    torch.Size([1322])
    torch.Size([5288])
    '''

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    # 创建测试数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

#=========3分类问题=========
    model = model_dict[args.model](X_train.shape, 3).to(device)
    print(model)


    # ====== 类别权重计算 ======
    class_counts = torch.tensor([629, 401, 229], dtype=torch.float)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()  # 归一化（可选）
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)


    epochs = args.epoch

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=epochs//3, gamma=0.5)  # 每epochs//3个 epoch 学习率减小为原来的50%

    # 定义列表来保存训练过程中的损失和准确率
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []

    print('\n==================================================   【训练】   ===================================================\n')
    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     criterion=criterion,
                                     data_loader=test_loader,
                                     device=device,
                                     epoch=epoch)
        # 保存训练过程中的损失和准确率
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        # 保存测试过程中的损失和准确率
        test_loss_all.append(val_loss)
        test_acc_all.append(val_acc)

    print("over")


    # 绘制训练过程中的损失和准确率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_loss_all, label='Test Loss')
    plt.plot(range(1, epochs + 1), train_acc_all, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_acc_all, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('value')
    plt.legend()
    plt.show()

    print(
        '\n==================================================   【测试评估指标】   ===================================================\n')

    # 模型评估阶段不需要梯度
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

#=========3分类标签评估指标调整========
    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"准确率 Accuracy: {acc:.4f}")
    print(f"精确率 Precision: {precision:.4f}")
    print(f"召回率 Recall: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1','Class 2'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

    # ========================== 推理时间统计 ==========================
    with torch.no_grad():
        # 单样本推理时间（取一个样本多次前向传播）
        single_sample = X_test[0].unsqueeze(0)  # shape: [1, 1, series, modal]
        repetitions = 100
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(repetitions):
            _ = model(single_sample)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        avg_sample_time = (end_time - start_time) / repetitions
        print(f"单样本平均推理时间: {avg_sample_time:.4f} 秒")

        # 整个测试集一轮推理时间
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for batch_x, _ in test_loader:
            _ = model(batch_x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        total_test_time = end_time - start_time
        print(f"整个测试集一轮推理时间: {total_test_time:.4f} 秒（总样本数: {len(test_dataset)}）")


if __name__ == '__main__':
    model_dict = {
        'mamba':mamba.Mamba
    }
    opt = parse_args()
    main(opt)













