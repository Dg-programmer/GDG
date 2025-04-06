from setuptools import setup, find_packages

setup(
    name='raga_detector',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'librosa',
        'numpy',
        'tensorflow',
        'yt-dlp',
        'soundfile',
        'scikit-learn',
        'matplotlib',
        'pydub'
    ],
    description='Detect Indian classical ragas from YouTube or audio files.',
    author='Your Name',
    author_email='your.email@example.com',
    python_requires='>=3.7',
)
