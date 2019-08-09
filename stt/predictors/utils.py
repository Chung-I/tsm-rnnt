import librosa
import soundfile as sf
import numpy as np
from stt.predictors import hyperparams as hp


def log_melfilterbank(y, amin=1e-8,
                      n_fft=hp.n_fft,
                      hop_length=hp.hop_length,
                      win_length=hp.win_length):
    mel_spec = librosa.feature.melspectrogram(y,
                                              sr=hp.sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=hp.n_mels)
    return np.log(np.maximum(amin, mel_spec))


def read_audio(path, sr=16000):
    y, orig_sr = sf.read(str(path))
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    return y, sr


def wavfile_to_feature(wavfile: str):

    y, _ = read_audio(wavfile, sr=16000)
    y, _ = librosa.effects.trim(y)
    feat = log_melfilterbank(y)
    return feat
