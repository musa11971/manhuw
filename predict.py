import os
import sys
import joblib
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from shared.manhuw import extract_features, load, preprocess_audio

def record_audio(duration, fs=22050):
    print(f'Recording for {duration} seconds...')
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording

def save_recording(recording, filename, fs=22050):
    wav.write(filename, fs, recording)

# Load the model through joblib
model_filename = input('Enter model filename: ')
model_path = os.path.join(os.getcwd(), model_filename)

if not os.path.isfile(model_path):
    sys.exit('Invalid model file')

model = joblib.load(model_path)
print('Loaded model')

# Record and save the recording
recording_duration = 20
sample_rate = 22050
recording_filename='recording.wav'

recording = record_audio(duration=recording_duration, fs=sample_rate)
save_recording(recording, filename=recording_filename, fs=sample_rate)
print('Stop recording...')

# Load recording
print('Loading recording...')
audio_data, sr = load(recording_filename)

# Preprocess data
print('Preprocessing recording...')
audio_data, sr = preprocess_audio(audio_data, sr)
save_recording(audio_data, filename=recording_filename, fs=sr)

# Extract features from recording
print('Extracting features...')
features = extract_features(audio_data, sr)

# Reshape the features array
features = np.array(features).reshape(1, -1)

# Predict using the model
print('Predicting...')
prediction = model.predict(features)
print(f"Predicted Reciter: {prediction[0]}")
print(prediction)