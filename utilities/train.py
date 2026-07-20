import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import numpy as np
import copy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def train_model(model, X_train, y_train, X_val, y_val,
                batch_size=config.BATCH_SIZE,
                lr=config.LEARNING_RATE,
                num_epochs=config.NUM_EPOCHS,
                weight_decay=config.WEIGHT_DECAY,
                patience=config.PATIENCE):
    """
    训练模型，使用验证集监控训练过程，支持 early stopping

    参数:
        model: nn.Module — 模型实例
        X_train: np.ndarray, shape (n_samples, n_channels, n_timepoints)
        y_train: np.ndarray, shape (n_samples,)
        X_val: np.ndarray, shape (n_samples, n_channels, n_timepoints)
        y_val: np.ndarray, shape (n_samples,)
        batch_size: 批大小
        lr: 学习率
        num_epochs: 最大训练轮数
        weight_decay: L2正则化系数
        patience: 验证集 acc 不提升时提前终止的等待轮数

    返回:
        model: 训练完成的模型（验证集 acc 最佳时的权重）
        history: dict — 包含训练/验证的 loss 和 accuracy 记录
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=batch_size, shuffle=False)

    # 计算类别权重，解决数据不平衡
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.STEP_SIZE,
                                                gamma=config.GAMMA)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping 逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]  "
                  f"Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  "
                  f"Val_Loss: {val_loss:.4f}  Val_Acc: {val_acc:.4f}"
                  f"  {'(best)' if val_acc == best_val_acc else ''}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(验证集 acc {patience} 轮未提升)")
            break

    # 恢复最佳模型权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复最佳模型权重 (Val_Acc: {best_val_acc:.4f})")

    print(f"\n训练完成 | 验证集最佳准确率: {best_val_acc:.4f}")

    return model, history
