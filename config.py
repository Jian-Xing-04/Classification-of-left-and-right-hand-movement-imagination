from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


EPOC_DATA_DIR_NEW = PROJECT_ROOT / "data" / "EPOC_Dataset"

# ==================== EPOC New Dataset ====================
EPOC_NEW_LABEL_MAP = {
    'right_hand': 0,
    'right_leg': 1,
    'left_hand': 2,
    'left_leg': 3,
    'eye_blink': 4,
    'break': 5,
}

# ==================== EDF Dataset ====================
EDF_DATA_DIR = PROJECT_ROOT / "data" / "after_data"

EDF_LABEL_MAP = {
    'left': 2,
    'right': 3,
    'break': 4,
}

# ==================== EPOC 3-type Dataset ====================
EPOC_DATA_DIR = PROJECT_ROOT / "data" / "EPOC_3type"

EPOC_EEG_CHANNELS = [
    'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7',
    'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8',
    'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4'
]

EPOC_CH_NAMES = [ch.replace('EEG.', '') for ch in EPOC_EEG_CHANNELS]

EPOC_SFREQ = 128.0

# EPOC 标签映射 (从文件名解析)
EPOC_LABEL_MAP = {
    'left': 0,
    'right': 1,
    'break': 2,
}

# 分割窗口长度 (秒) 和步长 (秒)
EPOC_WINDOW_SEC = 2.0
EPOC_STEP_SEC = 1.0

# ==================== Preprocessing ====================
BANDPASS_FREQ = (0.5, 45.0)
NOTCH_FREQ = 50.0
IIR_ORDER = 4
BANDPASS_sFREQ=(4,50)

# ==================== EEGNet Model ====================
EEGNET_F1 = 16         # 时序卷积滤波器数量 (8→16)
EEGNET_D = 2           # 深度乘数
EEGNET_F2 = 32         # 可分离卷积滤波器数量 (16→32, = F1*D)
EEGNET_KERNEL1 = 64    # 时序卷积核大小
EEGNET_DROPOUT = 0.5   # Dropout 率

# 从数据配置推导
EEGNET_N_CHANNELS = len(EPOC_EEG_CHANNELS)
EEGNET_N_TIMEPOINTS = int(EPOC_WINDOW_SEC * EPOC_SFREQ)
EEGNET_N_CLASSES = len(EPOC_LABEL_MAP)

# ==================== Training ====================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
WEIGHT_DECAY = 1e-4
STEP_SIZE = 30
GAMMA = 0.5
PATIENCE = 20  # early stopping 耐心值