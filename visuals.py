import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import soundfile as sf

def plot_wavefrom(y, sr,start,end):
    """
    Plot waveform, mel spectrogram, and MFCCs for a given audio signal.
    """
    #plt.figure(figsize=(12, 8))
    data,sr=sf.read(y)
    y=np.array(data)
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]


    # Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform of Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    #librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.xlim(start, end)           # seconds
   # frequency in Hz (optional)

    plt.show()
def plot_Spectogram(y,sr,start,end):
    # Mel Spectrogram
    data,sr=sf.read(y)
    y=np.array(data)
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]
     
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlim(start, end)
    plt.show()

def plot_MFCC(y,sr,start,end):
    # Mel Spectrogram
    data,sr=sf.read(y)
    y=np.array(data)
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]
 
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlim(start, end)
    plt.show()
def plot_Chroma(y,sr,start,end):
    data,sr=sf.read(y)
    y=np.array(data)
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]

   #Chroma Features 
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title('Chroma Features')
    plt.xlim(start, end)
    plt.show()
    #plt.xlim(start, end) 
def plot_Tempo(y,sr,start,end):
    data,sr=sf.read(y)
    y=np.array(data)
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]


    #Temop
    # Compute Tempo & Beat Frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Ensure tempo is defined and converted to a scalar float
    if isinstance(tempo, np.ndarray):  
        tempo = float(tempo[0])  # Extract first value if it's an array
    else:
        tempo = float(tempo)  # Direct conversion if it's already a single value

    # Plot Beat Tracking
    plt.figure(figsize=(10, 4))
    
    plt.plot(librosa.frames_to_time(beat_frames, sr=sr), np.ones_like(beat_frames), 'ro')
    plt.title(f'Tempo: {tempo:.2f} BPM (Beats Per Minute)')
    plt.xlabel('Time (seconds)')
    plt.xlim(start, end) 
    

    plt.show()
def plot_Pitch(y,sr,start,end):
    data,sr=sf.read(y)
    y=np.array(data)
   
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]



    #Pitch
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=1200)
    f0 = np.nan_to_num(f0)  # Replace NaNs with 0

    plt.figure(figsize=(12, 4))
    plt.plot(librosa.times_like(f0), f0, label="F0", color='r')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour (F0)")
    plt.xlim(start, end) 
    plt.legend()
    plt.show()


def plot_Chord(y,sr,start,end):
    data,sr=sf.read(y)
    y=np.array(data)
    # 
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]


    #CHrod
    harmonic, percussive = librosa.effects.hpss(y)
    chroma_harmonic = librosa.feature.chroma_cqt(y=harmonic, sr=sr)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(chroma_harmonic, sr=sr, x_axis="time", y_axis="chroma")
    plt.colorbar()
    #plt.xlim(start, end) 
    plt.title("Harmonic Chroma Features (Chord Recognition)")
    plt.xlim(start, end) 
    plt.show()
    #Music and speech recognition
def plot_Muspech(y,sr,start,end):
    data,sr=sf.read(y)
    y=np.array(data)
    if len(y.shape) > 1:
        y = y[:, 0]

        # Just take the first 30 seconds (30 * sample_rate samples)
        y= y[start*30:sr *end]

    harmonic, percussive = librosa.effects.hpss(y)
   
    plt.figure(figsize=(12, 4))
    plt.plot(harmonic, label="Harmonic (Music)")
    plt.plot(percussive, label="Percussive (Speech/Drums)", alpha=0.7)
    plt.xlim(start, end) 
    plt.legend()
    plt.title("Speech & Music Separation")
    plt.show()







