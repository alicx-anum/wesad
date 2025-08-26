import os
import pickle
import numpy as np
from scipy.signal import resample
import pandas as pd
from tqdm import tqdm

#数据预处理——对chest的ecg
'''
原先的700hz降采样到70hz
滑动窗口10s
步长5s，重叠率50%
'''

# ========== 配置 ==========
DATA_PATH = r"C:\xly\WESAD_Emotion_Recognition\WESAD"  # WESAD 数据集路径
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]  # 受试者编号，排除S12
TARGET_FS = 70         # 目标采样率（降采样到70Hz）
WINDOW_SEC = 10        # 窗口长度（秒）
STEP_SEC = 5           # 窗口步长（秒）
SAVE_CSV = "./wesad_ecg_3class.csv"

# ========================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def resample_signal(sig, orig_fs, target_fs):
    """降采样到目标采样率"""
    if orig_fs == target_fs:
        return sig
    num = int(len(sig) * target_fs / orig_fs)
    return resample(sig, num)

def sliding_window(sig, labels, win_size, step):
    """按滑动窗口切片"""
    X, y = [], []
    for start in range(0, len(sig) - win_size + 1, step):
        end = start + win_size
        seg = sig[start:end]       # (win_size,) 一维信号片段
        seg_labels = labels[start:end]
        label_mode = np.bincount(seg_labels).argmax()  # 该窗口内出现次数最多的标签
        X.append(seg)
        y.append(label_mode)
    return np.array(X), np.array(y)

def zscore_1d(x):
    """一维信号做 z-score 标准化"""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

all_rows = []

for subj in tqdm(SUBJECTS, desc="Processing subjects"):
    pkl_path = os.path.join(DATA_PATH, subj, subj + ".pkl")
    if not os.path.exists(pkl_path):
        print(f"跳过 {subj}, 文件不存在")
        continue

    data = load_pkl(pkl_path)

    # 取胸部ECG单通道信号，拉平成一维数组
    ecg = np.array(data["signal"]["chest"]["ECG"]).flatten()
    labels = np.array(data["label"])

    # 降采样：原始700Hz -> 目标70Hz
    ecg = resample_signal(ecg, 700, TARGET_FS)
    labels = resample_signal(labels.astype(float), 700, TARGET_FS)
    labels = np.round(labels).astype(int)  # 重采样后标签四舍五入为整数

    # 过滤只保留标签1,2,3（Baseline, Stress, Amusement）
    mask = np.isin(labels, [1, 2, 3])
    ecg = ecg[mask]
    labels = labels[mask]

    # 设置滑动窗口大小和步长（单位：采样点数）
    win_size = WINDOW_SEC * TARGET_FS
    step = STEP_SEC * TARGET_FS

    # 按滑动窗口切分信号和标签
    X_win, y_win = sliding_window(ecg, labels, win_size, step)

    # 对每个窗口的信号做标准化，整理数据和标签
    for sig, lab in zip(X_win, y_win):
        sig = zscore_1d(sig)  # 标准化
        lab = lab - 1         # 标签映射为0,1,2
        row = np.concatenate([sig, [lab]])  # 拼接信号数据和标签
        all_rows.append(row)

# 保存为CSV，数据列数 = 窗口长度 + 1（标签）
df = pd.DataFrame(all_rows)
df.to_csv(SAVE_CSV, index=False, header=False)
print(f"保存完成: {SAVE_CSV}, 样本数: {len(all_rows)}, 每行长度: {df.shape[1]}")
