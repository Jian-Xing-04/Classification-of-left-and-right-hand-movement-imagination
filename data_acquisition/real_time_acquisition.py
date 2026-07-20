import pylsl
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# ========== 配置参数 ==========
STREAM_NAME = 'openvibeSignal'  # OpenViBE LSL Export 默认流名称
BUFFER_SECONDS = 5  # 显示最近多少秒的数据
UPDATE_INTERVAL = 0.05  # 更新间隔（秒）
# ==============================

# 1. 连接LSL流
print(f"正在查找名为 '{STREAM_NAME}' 的LSL流...")
streams = pylsl.resolve_streams()
streams = [s for s in streams if s.name() == STREAM_NAME]

if not streams:
    raise RuntimeError(f"未找到名为 '{STREAM_NAME}' 的LSL流，请检查OpenViBE是否正在发送数据。")

inlet = pylsl.StreamInlet(streams[0])
info = inlet.info()
channel_count = info.channel_count()
sfreq = info.nominal_srate()

print(f"连接成功！通道数: {channel_count}, 采样率: {sfreq:.1f} Hz")

# 2. 准备环形缓冲区
buffer_length = int(BUFFER_SECONDS * sfreq)
buffers = [deque(maxlen=buffer_length) for _ in range(channel_count)]
ts_buffer = deque(maxlen=buffer_length)

# ... existing code ...

# 3. 初始化绘图
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
axes[-1].set_xlabel('时间 (秒)')
fig.suptitle('实时脑电信号', fontsize=14, y=1.01)
fig.tight_layout()

print("开始实时显示脑波数据... (按 Ctrl+C 退出)")

# 4. 主循环
try:
    start_time = None
    while True:
        samples, timestamps = inlet.pull_chunk(timeout=0.1)

        if samples:
            if start_time is None:
                start_time = timestamps[0]

            for sample, ts in zip(samples, timestamps):
                for ch_idx, val in enumerate(sample):
                    buffers[ch_idx].append(val)
                ts_buffer.append(ts - start_time)

            if len(ts_buffer) > 1:
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

                axes[0].set_xlim(max(0, x_data[-1] - BUFFER_SECONDS), x_data[-1])

                fig.canvas.draw()
                fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\n停止显示。")
finally:
    plt.ioff()
    plt.show()

