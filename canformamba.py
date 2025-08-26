import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from models import mamba      # 请确保这里导入你的Mamba模型
from utils import train_one_epoch, evaluate  # 你原来utils里函数

def load_data(device):
    data = pd.read_csv('./WESAD/wesad_ecg_3class.csv', header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, X_train.shape, y_train.unique()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_shape, unique_classes = load_data(device)

    seq_len = train_shape[2]
    in_channels = train_shape[3]
    num_classes = len(unique_classes)

    param_grid = {
        'dim': [64, 128],
        'depth': [4, 6],
        'hidden_dim': [128, 256],
        'dropout': [0.0]  # Mamba模型暂不支持dropout，先写0
    }

    best_acc = 0
    best_params = None

    for dim, depth, hidden_dim, dropout in itertools.product(
            param_grid['dim'], param_grid['depth'], param_grid['hidden_dim'], param_grid['dropout']):
        print(f"\n=== Training with dim={dim}, depth={depth}, hidden_dim={hidden_dim}, dropout={dropout} ===")

        model = mamba.Mamba(input_len=seq_len,
                      in_channels=in_channels,
                      num_classes=num_classes,
                      dim=dim,
                      depth=depth,
                      hidden_dim=hidden_dim).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

        epochs = 15
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss, val_acc = evaluate(model, val_loader, device, epoch)
            print(f"Epoch {epoch+1}/{epochs} - train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = {'dim': dim, 'depth': depth, 'hidden_dim': hidden_dim, 'dropout': dropout}
            torch.save(model.state_dict(), "best_mamba.pth")
            print(f"New best val acc: {best_acc:.4f}")

    print("\n=== Best hyperparameters ===")
    print(best_params)
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
