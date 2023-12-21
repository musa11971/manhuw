import os
import librosa
import numpy as np

train_data_dir = 'audio'

def extract_features(file_name):
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

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

    return mfccs, chroma, mel, contrast, tonnetz

# Get the list of all subdirectories in the training data directory
sub_folders = [name for name in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, name))]
print(f'Found {len(sub_folders)} reciters')

# Loop through each subdirectory
for sub_folder in sub_folders:
    print(f'starting {sub_folder}')
    sub_folder_full = os.path.join(train_data_dir, sub_folder)
    files = os.listdir(sub_folder_full)
    files_mp3 = [file for file in files if file.endswith('.mp3')]

    # Loop through each .mp3 file
    for file_mp3 in files_mp3:
        print(f'├─ {file_mp3}')
        file_mp3_full = os.path.join(train_data_dir, sub_folder, file_mp3)

        # Extract features
        features = extract_features(file_mp3_full)

    print('')