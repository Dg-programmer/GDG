import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_model(path='data/raga_detection_model.h5'):
    try:
        return keras.models.load_model(path)
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

def load_scaler():
    try:
        scaler = StandardScaler()
        scaler.mean_ = np.load('data/scaler_mean.npy')
        scaler.scale_ = np.load('data/scaler_scale.npy')
        scaler.n_features_in_ = np.load('data/scaler_n_features.npy').item()
        scaler.var_ = np.load('data/scaler_var.npy')
        scaler.n_samples_seen_ = np.load('data/scaler_n_samples_seen.npy').item()
        return scaler
    except Exception as e:
        raise RuntimeError(f"Error loading scaler: {e}")

def load_label_encoder():
    try:
        le = LabelEncoder()
        le.classes_ = np.load('data/label_encoder_classes.npy', allow_pickle=True)
        return le
    except Exception as e:
        raise RuntimeError(f"Error loading label encoder: {e}")

# Load once and reuse
model = load_model()
scaler = load_scaler()
label_encoder = load_label_encoder()
