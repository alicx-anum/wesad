import itertools
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models import PHemo  # 假设你PHemo里有H2SingleBVP
import time

# 你数据读取和处理部分不变，这里简写
def load_data(device):
    data = pd.read_csv('./WESAD/wesad_ecg_3class.csv', header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.int64).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.int64).to(device)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset, X_train.shape[2]

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_num = 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_num += batch_x.size(0)
    return total_loss / total_num, total_correct / total_num

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_num = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_num += batch_x.size(0)
    return total_loss / total_num, total_correct / total_num

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset, test_dataset, feature_len = load_data(device)
    batch_size = 64
    epochs = 30
    lr = 5e-5

    # 参数网格
    dropout_rates = [0.3, 0.5, 0.7]
    units_list = [64, 128, 256]
    n_list = [2, 4, 6]

    results = []

    for dropout_rate, units, n in itertools.product(dropout_rates, units_list, n_list):
        print(f"Training with dropout={dropout_rate}, units={units}, n={n}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = PHemo.H2SingleBVP(dropout_rate=dropout_rate, units=units, n=n, in_features=feature_len, num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

        best_test_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f} Test Acc={test_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc

        results.append({
            "dropout_rate": dropout_rate,
            "units": units,
            "n": n,
            "best_test_acc": best_test_acc
        })

    # 保存结果
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.sort_values(by="best_test_acc", ascending=False)
    print(df)
    df.to_csv("grid_search_results.csv", index=False)

if __name__ == "__main__":
    main()
