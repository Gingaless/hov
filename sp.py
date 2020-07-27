# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import librosa
import librosa.display
import sounddevice as sd
import json


def save_stft_json(S, save_path, ori_sr, stft_sr):
    
    data = {'audio' : S, 'ori_sr' : 44100, 'stft_sr' : 8000}
    with open(save_path, 'w') as f:
        json.dump(data, f)
        f.close()
    return

def load_stft_json(path):
    
    assert os.path.exists(path)
    f = open(path)
    data = json.load(f)
    return data['audio'], data['ori_sr'], data['stft_sr']


def audio_file_to_mel(ori, save_dir, save_name, ori_sr=44100, mel_sr=8000, **kwargs):
    
    assert os.path.exists(ori)
    
    ado, _ = librosa.core.load(ori, ori_sr)
    ado = librosa.core.resample(ado, ori_sr, mel_sr)
    melspec = librosa.feature.melspectrogram(ado, mel_sr, **kwargs)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    full_name = os.path.join(save_dir, save_name)
    np.save(full_name, melspec)
    return np.shape(melspec)


def audio_file_to_stft(ori, save_dir, save_name, ori_sr=44100, stft_sr=8000, **kwargs):
    
    assert os.path.exists(ori)
    
    ado, _ = librosa.load(ori, ori_sr)
    ado = librosa.core.resample(ado, ori_sr, target_sr)
    S = librosa.core.stft(ado, target_sr, **kwargs)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    full_name = os.path.join(save_dir, save_name)
    np.save(full_name, S)
    return np.shape(S)


def melspec_to_audio(melspec, ori_sr=44100, mel_sr=8000, **kwargs):
    
    ado = librosa.feature.inverse.mel_to_audio(melspec, sr=mel_sr, **kwargs)
    resampled = librosa.core.resample(ado, mel_sr, ori_sr)
    return resampled


def mel_file_to_audio(ori_path, ori_sr=44100, mel_sr=8000, **kwargs):
    
    assert os.path.exists(ori_path)
    
    melspec = np.load(ori_path)
    return melspec_to_audio(melspec, ori_sr=ori_sr, mel_sr=mel_sr, **kwargs)


def stft_file_to_audio(ori_path, ori_sr=44100, stft_sr=8000, **kwargs):
    
    assert os.path.exists(ori_path)
    
    S = np.load(ori_path)
    ado = librosa.core.istft(S, **kwargs)
    resampled = librosa.core.resample(ado, stft_sr, ori_sr)
    return resampled



# %%
