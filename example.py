from raga_detector import predict_raga_from_file
from raga_detector.core import extract_features
from raga_detector.visuals import  plot_Chord,plot_Tempo,plot_wavefrom,plot_Chroma,plot_MFCC,plot_Muspech,plot_Spectogram,plot_Pitch
from raga_detector.model_loader import model,label_encoder,scaler
#stadnardx,y,z=extract_features("data/WvUt17hE4YA.wav",None)
z= [
    -210.5, 115.2, 6.8, -3.1, 1.5, -0.7, 0.3,
    0.0, -0.1, 0.2, 0.1, -0.05, 0.02,

    # Extra features (5)l 
    0.04,       # Zero Crossing Rate
    2100.0,     # Spectral Bandwidth
    3400.5,     # Spectral Rolloff
    0.015,      # RMS Energy
    0.12        
]
#raga = predict_raga_from_file(0.47,2450.83,z)
#print("Predicted Raga:", raga)
#plot_audio_features()
x,y,z=extract_features("mi.mp3",None)
g=predict_raga_from_file(x,y,z)
print(g)
#plot_wavefrom("mi.mp3",None,20,40)
plot_MFCC("mi.mp3",None,20,70)