# 工具包用来集成训练和测试的代码
import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F

#针对mamba中class2容易被判断为class0，调整权重

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_class_weights(train_loader, smooth="sqrt", device="cpu"):
    """
    根据 train_loader 统计类别数量并生成平滑权重
    smooth: None | "sqrt" | "cbrt"
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    counts = torch.bincount(torch.tensor(all_labels))
    counts = counts.float()

    if smooth == "sqrt":
        weights = 1.0 / torch.sqrt(counts)
    elif smooth == "cbrt":
        weights = 1.0 / torch.pow(counts, 1/3)
    else:
        weights = 1.0 / counts

    weights = weights / weights.sum()  # 归一化
    return weights.to(device)


def get_loss_function(train_loader, loss_type="ce", smooth="sqrt", gamma=2.0, device="cpu"):
    """
    loss_type: "ce" (CrossEntropyLoss) | "focal" (FocalLoss)
    smooth: None | "sqrt" | "cbrt"
    """
    weights = get_class_weights(train_loader, smooth=smooth, device=device)

    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_type == "focal":
        return FocalLoss(alpha=weights, gamma=gamma)
    else:
        raise ValueError("loss_type 必须是 'ce' 或 'focal'")

def train_one_epoch(model, optimizer, criterion, data_loader, device, epoch):
    scaler = GradScaler()
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        with autocast():
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))  # 用传入的 criterion

        pred_classes = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch+1, accu_loss.item() / (step + 1), accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch):
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        with autocast():
            pred = model(images.to(device))
            loss = criterion(pred, labels.to(device))  # 用传入的 criterion

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch+1, accu_loss.item() / (step + 1), accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# def train_one_epoch(model, optimizer, criterion,data_loader, device, epoch):
#     scaler = GradScaler()  # 在训练最开始之前实例化一个GradScaler对象
#     model.train()
#     #加权重后，修改loss计算
#     loss = criterion(outputs, labels.to(device))
#
#     #loss_function = torch.nn.CrossEntropyLoss()
#     accu_loss = torch.zeros(1).to(device)  # 累计损失 {Tensor:(1,)} tensor([0.])
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数 {Tensor:(1,)} tensor([0.])
#     optimizer.zero_grad()
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]
#
#         # 前向过程(model + loss)开启 autocast，混合精度训练
#         with autocast():
#             outputs = model(images.to(device))
#             loss = loss_function(outputs, labels.to(device))
#
#         # outputs = model(images.to(device))
#         pred_classes = torch.max(outputs, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         # loss = loss_function(outputs, labels.to(device))
#         # loss.backward()
#         accu_loss += loss.detach()
#
#         data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
#                                                                                accu_loss.item() / (step + 1),
#                                                                                accu_num.item() / sample_num)
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#
#         # optimizer.step()
#         # optimizer.zero_grad()
#
#         optimizer.zero_grad()  # 梯度清零
#         scaler.scale(loss).backward()  # 梯度放大
#         scaler.step(optimizer)  # unscale梯度值
#         scaler.update()
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num
#
#
# @torch.no_grad()
# def evaluate(model, data_loader, device, epoch):
#     loss_function = torch.nn.CrossEntropyLoss()
#
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]
#         with autocast():
#             pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         loss = loss_function(pred, labels.to(device))
#         accu_loss += loss
#
#         data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
#                                                                                accu_loss.item() / (step + 1),
#                                                                                accu_num.item() / sample_num)
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num
















