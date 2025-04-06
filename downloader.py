import yt_dlp

def download_audio_from_youtube(youtube_url):
    """
    Downloads the audio from a YouTube URL and saves it as a WAV file.

    Returns the filename if successful, otherwise None.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'wav',
        'outtmpl': '%(id)s.wav',  # Saves as <video_id>.wav
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_id = info_dict.get('id', None)
            if video_id:
                return f"{video_id}.wav"
            else:
                return None
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None
