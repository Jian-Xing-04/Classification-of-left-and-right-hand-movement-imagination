import pylsl
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utilities.preprocess import preprocess_eeg_data
from models.EEGNet import EEGNet

STREAM_NAME = 'openvibeSignal'
WINDOW_SEC = config.EPOC_WINDOW_SEC
STEP_SEC = config.EPOC_STEP_SEC
MODEL_PATH = config.PROJECT_ROOT / "saved_models" / "eegnet_best.pth"
LABEL_NAMES = list(config.EPOC_LABEL_MAP.keys())


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EEGNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def connect_lsl_stream(stream_name):
    print(f"正在查找名为 '{stream_name}' 的LSL流...")
    streams = pylsl.resolve_streams()
    streams = [s for s in streams if s.name() == stream_name]
    if not streams:
        raise RuntimeError(f"未找到名为 '{stream_name}' 的LSL流，请检查OpenViBE是否正在发送数据。")
    inlet = pylsl.StreamInlet(streams[0])
    info = inlet.info()
    channel_count = info.channel_count()
    sfreq = info.nominal_srate()
    print(f"连接成功！通道数: {channel_count}, 采样率: {sfreq:.1f} Hz")
    return inlet, channel_count, sfreq


def init_plot(channel_count, buffer_seconds):
    plt.ion()
    fig, axes = plt.subplots(channel_count, 1, figsize=(12, max(8, channel_count * 1.2)), sharex=True)
    if channel_count == 1:
        axes = [axes]

    lines = []
    colors = plt.cm.tab10(np.linspace(0, 1, channel_count))
    for ch_idx, ax in enumerate(axes):
        line, = ax.plot([], [], color=colors[ch_idx % len(colors)], linewidth=0.8)
        lines.append(line)
        ax.set_ylabel(f'Ch{ch_idx + 1}', rotation=0, ha='right', va='center', fontsize=9)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Real-time EEG Acquisition + Classification', fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes, lines


def update_plot(axes, lines, buffers, ts_buffer, channel_count, buffer_seconds):
    x_data = np.array(ts_buffer)
    for ch_idx in range(channel_count):
        y_data = np.array(buffers[ch_idx])
        if len(y_data) > len(x_data):
            y_data = y_data[-len(x_data):]
        elif len(y_data) < len(x_data):
            y_data = np.pad(y_data, (0, len(x_data) - len(y_data)), constant_values=np.nan)
        lines[ch_idx].set_data(x_data, y_data)
        axes[ch_idx].relim()
        axes[ch_idx].autoscale_view(scalex=False)
    axes[0].set_xlim(max(0, x_data[-1] - buffer_seconds), x_data[-1])


def run_inference(model, device, buffers, channel_count, n_timepoints, label_names):
    data_window = np.zeros((channel_count, n_timepoints))
    for ch_idx in range(channel_count):
        data_window[ch_idx] = np.array(buffers[ch_idx])[-n_timepoints:]

    X_input = data_window[np.newaxis, :, :]
    X_processed = preprocess_eeg_data(X_input)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        output = model(X_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = probs[pred_idx]

    pred_label = label_names[pred_idx]
    prob_str = ' | '.join([f'{label_names[i]}: {probs[i]:.1%}' for i in range(len(label_names))])
    print(f"预测: {pred_label.upper()} (置信度: {confidence:.1%}) | {prob_str}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"加载模型... 设备: {device}")
    model = load_model(MODEL_PATH, device)
    print("模型加载完成")

    inlet, channel_count, sfreq = connect_lsl_stream(STREAM_NAME)

    n_timepoints = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)
    buffer_size = n_timepoints + step_samples
    buffers = [deque(maxlen=buffer_size) for _ in range(channel_count)]
    ts_buffer = deque(maxlen=buffer_size)

    fig, axes, lines = init_plot(channel_count, WINDOW_SEC * 2)
    print(f"开始实时采集与推理... (窗口={WINDOW_SEC}s, 步长={STEP_SEC}s, 按 Ctrl+C 退出)")

    try:
        start_time = None
        sample_counter = 0
        last_infer_counter = 0
        inference_count = 0

        while True:
            samples, timestamps = inlet.pull_chunk(timeout=0.1)

            if samples:
                if start_time is None:
                    start_time = timestamps[0]

                for sample, ts in zip(samples, timestamps):
                    for ch_idx, val in enumerate(sample):
                        buffers[ch_idx].append(val)
                    ts_buffer.append(ts - start_time)
                    sample_counter += 1

                if len(ts_buffer) > 1:
                    update_plot(axes, lines, buffers, ts_buffer, channel_count, WINDOW_SEC * 2)

                if sample_counter - last_infer_counter >= step_samples and sample_counter >= n_timepoints:
                    inference_count += 1
                    print(f"[推理 #{inference_count}] ", end='')
                    run_inference(model, device, buffers, channel_count, n_timepoints, LABEL_NAMES)
                    last_infer_counter = sample_counter

                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print(f"\n停止。共完成 {inference_count} 次推理。")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
