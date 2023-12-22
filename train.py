import os
import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from shared.manhuw import extract_features, load, preprocess_audio

train_data_dir = 'audio'

# Get the list of all subdirectories in the training data directory
sub_folders = [name for name in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, name))]
reciter_count = len(sub_folders)
print(f'Found {reciter_count} reciters')

features_list = []
labels = []

# Loop through each subdirectory
for index, sub_folder in enumerate(sub_folders):
    print(f'training with {sub_folder} {index+1}/{reciter_count}')
    sub_folder_full = os.path.join(train_data_dir, sub_folder)
    files = os.listdir(sub_folder_full)
    files_mp3 = [file for file in files if file.endswith('.mp3')]

    # Loop through each .mp3 file
    for file_mp3 in files_mp3:
        print(f'├─ {file_mp3}')
        file_mp3_full = os.path.join(train_data_dir, sub_folder, file_mp3)

        # Load file, with 5 seconds offset to avoid basmallah
        print('   loading...')
        audio_data, sr = load(file_mp3_full, skip_seconds=5)

        # Preprocess data
        print('   preprocessing...')
        audio_data, sr = preprocess_audio(audio_data, sr)

        # Extract features
        print('   extracting...')
        features = extract_features(audio_data, sr)
        features_list.append(features)
        labels.append(sub_folder)

    print('')

# Convert to a DataFrame
df = pd.DataFrame(features_list)
df['label'] = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['label'], test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save model
saved_model_filename = f'{time.time()}_trained_model.joblib'
joblib.dump(model, saved_model_filename)
print(f'Saved model to {saved_model_filename}')