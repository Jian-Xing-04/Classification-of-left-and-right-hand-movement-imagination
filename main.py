import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utilities.data_loader import load_epoc_dataset
from utilities.preprocess import preprocess_eeg_data
from models.EEGNet import EEGNet
from utilities.train import train_model


def main():
    # ==================== 1. 加载数据 ====================
    print("=" * 50)
    print("Step 1: 加载数据")
    print("=" * 50)
    X, y, label_map = load_epoc_dataset()
    print(f"标签映射: {label_map}")

    # ==================== 2. 预处理 ====================
    print(f"\n{'=' * 50}")
    print("Step 2: 预处理 (带通滤波 + 陷波滤波)")
    print("=" * 50)
    X = preprocess_eeg_data(X)
    print(f"预处理后数据形状: {X.shape}")

    # ==================== 3. 划分训练集/验证集 ====================
    print(f"\n{'=' * 50}")
    print("Step 3: 划分训练集/验证集/测试集 (70/15/15)")
    print("=" * 50)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # ==================== 4. 训练模型 ====================
    print(f"\n{'=' * 50}")
    print("Step 4: 训练 EEGNet")
    print("=" * 50)
    model = EEGNet()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {device}")

    model, history = train_model(model, X_train, y_train, X_val, y_val)

    # ==================== 5. 保存模型 ====================
    print(f"\n{'=' * 50}")
    print("Step 5: 保存模型")
    print("=" * 50)
    save_dir = config.PROJECT_ROOT / "saved_models"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "eegnet_best.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'label_map': label_map,
        'config': {
            'F1': config.EEGNET_F1,
            'D': config.EEGNET_D,
            'F2': config.EEGNET_F2,
            'n_channels': config.EEGNET_N_CHANNELS,
            'n_timepoints': config.EEGNET_N_TIMEPOINTS,
            'n_classes': config.EEGNET_N_CLASSES,
        }
    }, save_path)
    print(f"模型已保存至: {save_path}")

    # ==================== 6. 详细性能评估 ====================
    print(f"\n{'=' * 50}")
    print("Step 6: 测试集独立评估")
    print("=" * 50)

    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    accuracy = (y_pred == y_test).mean()
    print(f"\n测试集总体准确率: {accuracy:.4f}")

    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵:")
    print(f"{'':>10}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>10}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>10}", end="")
        print()

    print(f"\n训练曲线摘要:")
    print(f"  初始 Val_Acc: {history['val_acc'][0]:.4f}")
    print(f"  最终 Val_Acc: {history['val_acc'][-1]:.4f}")
    print(f"  最佳 Val_Acc: {max(history['val_acc']):.4f} (Epoch {np.argmax(history['val_acc']) + 1})")

if __name__ == "__main__":
    main()
