import numpy as np
import mne
import os
from pathlib import Path
from tqdm import tqdm
import config

EVENT_MAP = {
    'OVTK_StimulationId_Number_01': 'cue',
    'OVTK_StimulationId_Number_02': 'left',
    'OVTK_StimulationId_Number_03': 'right',
    'OVTK_StimulationId_Number_04': 'idle',
    'OVTK_StimulationId_Number_05': 'end',
}

LABEL_MAP = {
    'left': 0,
    'right': 1,
    'idle': 2,
}


def _segment_data(eeg_data, label, sfreq, window_sec, step_sec):
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


def _get_subject_name(filename):
    name = filename.replace('.edf', '')
    name = name.split(']')[-1].strip()
    return name if name else 'unknown'


def load_edf_dataset(data_dir=config.EDF_DATA_DIR,
                     sfreq=config.EPOC_SFREQ,
                     window_sec=2.0, step_sec=1.0,
                     labels=['left', 'right', 'idle'],
                     subject=None):

    data_dir = Path(data_dir)
    edf_files = sorted(data_dir.glob('*.edf'))

    if not edf_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到EDF文件")

    if subject:
        edf_files = [f for f in edf_files if _get_subject_name(f.name) == subject]
        print(f"筛选被试 '{subject}': {len(edf_files)} 个EDF文件")

    print(f"找到 {len(edf_files)} 个EDF文件")

    all_segments = []
    all_labels = []

    for edf_file in tqdm(edf_files, desc="加载EDF文件"):
        try:
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
        except Exception as e:
            print(f"  跳过 {edf_file.name}: 读取失败 - {e}")
            continue

        annotations = raw.annotations
        if len(annotations) == 0:
            print(f"  跳过 {edf_file.name}: 无事件标记")
            continue

        event_times = annotations.onset
        event_descs = annotations.description

        i = 0
        while i < len(event_descs):
            if event_descs[i] not in EVENT_MAP:
                i += 1
                continue

            event_type = EVENT_MAP[event_descs[i]]

            if event_type in labels:
                start_time = event_times[i]

                j = i + 1
                while j < len(event_descs):
                    if EVENT_MAP.get(event_descs[j]) == 'end':
                        end_time = event_times[j]
                        break
                    j += 1
                else:
                    end_time = start_time + 5.0

                onset_sample = int(start_time * sfreq)
                end_sample = int(end_time * sfreq)

                if onset_sample >= raw.get_data().shape[1]:
                    i += 1
                    continue

                end_sample = min(end_sample, raw.get_data().shape[1])

                if end_sample - onset_sample < window_sec * sfreq:
                    i += 1
                    continue

                eeg_data = raw.get_data()[:, onset_sample:end_sample]
                eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)

                numeric_label = LABEL_MAP[event_type]
                segs, lbls = _segment_data(eeg_data, numeric_label, sfreq, window_sec, step_sec)

                if len(segs) > 0:
                    all_segments.append(segs)
                    all_labels.append(lbls)

            i += 1

        raw.close()

    if not all_segments:
        raise RuntimeError("未能从EDF数据集中加载任何有效样本")

    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)

    label_map = {k: LABEL_MAP[k] for k in labels if k in LABEL_MAP}

    print(f"\nEDF数据集加载完成: {X.shape[0]} 个样本, "
          f"{X.shape[1]} 通道, {X.shape[2]} 时间点/段")
    print(f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, label_map


if __name__ == "__main__":
    X, y, label_map = load_edf_dataset()
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Label map: {label_map}")