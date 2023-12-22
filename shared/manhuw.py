import librosa
import numpy as np
import noisereduce as nr

default_sample_rate=22050

def load(file_name, skip_seconds=0):
    return librosa.load(file_name, sr=None, res_type='kaiser_fast')

def preprocess_audio(audio_data, rate):
    # Apply preprocessing steps
    audio_data = nr.reduce_noise(y=audio_data, sr=rate)
    audio_data = librosa.util.normalize(audio_data)
    audio_data, _ = librosa.effects.trim(audio_data)
    audio_data = librosa.resample(audio_data, orig_sr=rate, target_sr=default_sample_rate)
#     audio_data = fix_length(audio_data)
    rate = default_sample_rate

    return audio_data, rate

def extract_features(X, sample_rate):
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)

    # Concatenate all feature arrays into a single 1D array
    combined_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return combined_features