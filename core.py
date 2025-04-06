import yt_dlp
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import soundfile as sf
from pydub import AudioSegment
#from .core import extract_features
#from .downloader import download_audio_from_youtube
#from .model_loader import model, scaler, label_encoder
def extract_features(y, sr):
    try:
        data,sr=sf.read(y)
        y=np.array(data)
        #def extract_features(y, sr):
        # If stereo, convert to mono
# If stereo, convert to mono
        if len(y.shape) > 1:
            y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y = y[:sr * 30]


        if not isinstance(y, np.ndarray):
              print("ERROR: y is not a numpy array!", type(y))
              raise ValueError("A data must be of type numpy.ndarray")

        #print(type(y))
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=18)

        chroma_stft_mean = np.mean(chroma_stft) if chroma_stft.size > 0 else 0
        spec_cent_mean = np.mean(spec_cent) if spec_cent.size > 0 else 0
        mfcc_mean = np.mean(mfcc, axis=1) if mfcc.size > 0 else np.zeros(18)
        mfcc_features = mfcc_mean.flatten()

        return chroma_stft_mean, spec_cent_mean, mfcc_features
    except Exception as e: 
        print(f"Error extracting features: {e}")
        return None, None, None