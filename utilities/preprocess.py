import numpy as np
import mne
from pathlib import Path
from tqdm import tqdm
import config

def preprocess_eeg_data(X, sfreq=config.EPOC_SFREQ, bandpass=config.BANDPASS_FREQ, notch=config.NOTCH_FREQ):


    ch_names = [f'EEG{i:03d}' for i in range(X.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    X_processed = []

    for i in tqdm(range(X.shape[0]), desc="Preprocessing"):
        raw = mne.io.RawArray(X[i].copy(), info, verbose=False)

        raw.filter(bandpass[0], bandpass[1], method='iir',
                   iir_params=dict(order=config.IIR_ORDER, ftype='butter'),
                   verbose=False)
        raw.notch_filter(notch, method='iir',
                         iir_params=dict(order=config.IIR_ORDER, ftype='butter'),
                         verbose=False)

        X_processed.append(raw.get_data())

    return np.array(X_processed)
