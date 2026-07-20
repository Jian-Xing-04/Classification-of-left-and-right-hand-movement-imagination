import pandas as pd
import numpy as np
import mne
from mne.io import RawArray
from mne import create_info
from datetime import datetime
import os
import re
from pathlib import Path
import config


# ... existing code ...

# ==================== EPOC 3-type Dataset Loader ====================

def _parse_epoc_label(filename):
    """从文件名解析任务标签"""
    name_lower = filename.lower()
    if 'left' in name_lower:
        return 'left'
    elif 'right' in name_lower:
        return 'right'
    elif 'break' in name_lower:
        return 'break'
    else:
        raise ValueError(f"无法从文件名识别标签: {filename}")


def _segment_data(eeg_data, label, sfreq, window_sec, step_sec):
    """
    将连续EEG数据切分为固定长度的段，并生成对应标签

    返回:
        segments: np.ndarray, shape (n_segments, n_channels, n_timepoints)
        labels: np.ndarray, shape (n_segments,)
    """
    n_channels, n_samples = eeg_data.shape
    window_len = int(window_sec * sfreq)
    step_len = int(step_sec * sfreq)

    segments = []
    labels = []

    start = 0
    while start + window_len <= n_samples:
        seg = eeg_data[:, start:start + window_len]
        segments.append(seg)
        labels.append(label)
        start += step_len

    if not segments:
        return np.empty((0, n_channels, window_len)), np.empty((0,), dtype=int)

    return np.array(segments), np.array(labels)


def load_epoc_dataset(data_dir=config.EPOC_DATA_DIR,
                      eeg_channels=config.EPOC_EEG_CHANNELS, sfreq=config.EPOC_SFREQ,
                      window_sec=2.0, step_sec=1.0, labels=list(config.EPOC_LABEL_MAP.keys())):

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到CSV文件")

    print(f"找到 {len(csv_files)} 个CSV文件")

    all_segments = []
    all_labels = []

    for csv_file in csv_files:
        try:
            task_label = _parse_epoc_label(csv_file.name)
        except ValueError as e:
            print(f"跳过: {e}")
            continue

        if task_label not in labels:
            print(f"跳过 (标签不在目标列表中): {csv_file.name}")
            continue

        numeric_label = config.EPOC_LABEL_MAP[task_label]

        # 读取CSV，跳过第一行元数据
        df = pd.read_csv(csv_file, skiprows=1)

        # 提取可用通道
        available = [ch for ch in eeg_channels if ch in df.columns]
        if not available:
            print(f"警告: {csv_file.name} 中无有效EEG通道，跳过")
            continue

        eeg_data = df[available].values.T.astype(np.float64)
        eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)

        segs, lbls = _segment_data(eeg_data, numeric_label, sfreq, window_sec, step_sec)

        if len(segs) > 0:
            all_segments.append(segs)
            all_labels.append(lbls)
            print(f"  {csv_file.name} → 标签='{task_label}'({numeric_label}), "
                  f"切分出 {len(segs)} 段")
        else:
            print(f"  {csv_file.name} → 数据不足以切分，跳过")

    if not all_segments:
        raise RuntimeError("未能从数据集中加载任何有效样本")

    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)

    label_map = {k: config.EPOC_LABEL_MAP[k] for k in labels if k in config.EPOC_LABEL_MAP}

    print(f"\n数据集加载完成: {X.shape[0]} 个样本, "
          f"{X.shape[1]} 通道, {X.shape[2]} 时间点/段")
    print(f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, label_map


if __name__ == "__main__":
    # 测试 EPOC 数据集加载
    X, y, label_map = load_epoc_dataset()
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Label map: {label_map}")
