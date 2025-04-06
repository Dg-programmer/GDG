import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = remove INFO, 2 = remove WARNING, 3 = remove ERROR
warnings.filterwarnings('ignore')
import yt_dlp
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import soundfile as sf
from pydub import AudioSegment
from .core import extract_features
from .downloader import download_audio_from_youtube
from .model_loader import model, scaler, label_encoder



# Load the trained model

# Function to extract features from audio data

# Function to predict raga from extracted features
def predict_raga_from_file(chroma, spec_cent, mfcc):
    if chroma is None or spec_cent is None or mfcc is None:
        return "Error: Could not extract features"

    feature_vector = np.concatenate([np.array([chroma, spec_cent]), mfcc])
    feature_vector = feature_vector.reshape(1, -1)

    try:
        feature_vector = scaler.transform(feature_vector)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return "Error: Scaling failed"

    try:
        prediction = model.predict(feature_vector)
        predicted_class = np.argmax(prediction)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error: Prediction failed"

    try:
        le = LabelEncoder()
        le.classes_ = np.load('data/label_encoder_classes.npy', allow_pickle=True)
        raga = le.inverse_transform([predicted_class])[0]
    except Exception as e:
        print(f"Error loading label encoder or inverse transforming: {e}")
        return "Error: Label transformation failed"

    return raga

# Load the scaler


# Function to download audio from YouTube using yt-dlp
def download_audio_from_youtube(youtube_url):
    video_id = "WvUt17hE4YA"
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'data/WvUt17hE4YA.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'ffmpeg_location': 'D:\Downloads\ffmpeg-7.1.1-essentials_build.7z\ffmpeg-7.1.1-essentials_build\bin', # or the full path to ffmpeg.exe if needed
        'verbose': True
    

    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_id = info_dict.get('id', None)
            return f"{video_id}.wav"
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Main function to predict raga from YouTube link
def predict_raga_from_youtube(youtube_url):
    audio_file = download_audio_from_youtube(youtube_url)
    
    if not audio_file:
        return "Error: Could not download audio from YouTube"
    
    try:
        # Load audio using soundfile, then convert to numpy array
        data, sr = sf.read(audio_file)
        y = data[:, 0] if len(data.shape) > 1 else data  # Handle mono/stereo
        
        # Ensure sample rate is consistent
        librosa.resample(y, orig_sr=sr, target_sr=sr)
        
        # Limit duration to 30 seconds
        y = y[:sr*30]
            
        chroma, spec_cent, mfcc = extract_features(y, sr)
        
        if chroma is None or spec_cent is None or mfcc is None:
            return "Error: Could not extract features from YouTube audio"
        
        return predict_raga_from_file(chroma, spec_cent, mfcc)
    
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return "Error: Audio processing failed"

# Example usage
