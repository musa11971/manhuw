<p align="center"><img src=".github/icon.png" width="200"></p>

# manhuw
> manhuw (من هو)  
> _en: 'who is he?'_

Manhuw is a Quran Reciter Recognition app. It is a proof of concept for an application capable of recognizing and identifying different Quran reciters from audio recordings. It uses advanced audio processing and machine learning techniques to analyze recitations and match them to known reciters.

## Features
- **Voice Recognition**: Identify Quran reciters from audio clips.
- **Audio Preprocessing**: Includes noise reduction, normalization, and trimming to ensure consistent audio quality.
- **Machine Learning**: Utilizes a trained model to distinguish between different reciters' voices.

## Installation
```bash
# Clone the repository
git clone [repository-url]

# Navigate to the project directory
cd manhuw

# Install required Python packages
pip install -r requirements.txt
```

## Usage
### Training
By default, audio training data is included in the `audio` directory. Example:  

```
audio/
│
├── abdulbasit_abdulsamad/
│   ├── recording1.mp3
│   ├── recording2.mp3
│   └── ...
│
├── ahmed_talib_hameed/
│   ├── recording1.mp3
│   ├── recording2.mp3
│   └── ...
```
To train the model using these recordings, run:  
```bash
python3 train.py
```
This will train the model and output a `.joblib` model file.  
_(e.g. `1703241495.179929_trained_model.joblib`)_

### Predicting
```bash
python3 predict.py
> Enter model filename: 1703241495.179929_trained_model.joblib
> Loaded model
> Recording for 20 seconds...
> Stop recording...
> Loading recording...
> Preprocessing recording...
> Extracting features...
> Predicting...
> Predicted Reciter: muhammad_al_lohaidan
['muhammad_al_lohaidan']
```

### Contributing
All contributions are welcomed.

### License
[GNU General Public License v3.0](LICENSE.md). Manhuw is provided for the sole purpose of benefiting the Ummah and is intended to be used only for noble, Islamic and ethical purposes. No closed source usage allowed.
