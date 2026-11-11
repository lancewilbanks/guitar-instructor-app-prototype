# Audio.py -- audio extraction and processing utilities for guitar instructor project

import os
import numpy as np
from moviepy import VideoFileClip

audio_file_path = '/workspaces/dl_workspace/projects/guitar_instructor/uploads/heyjoe_snippet.mp4'

def extract_audio_to_numpy(resource_path, target_fps=44100):
    """
    Extracts audio from an audio/video file and returns it as a NumPy array.

    Parameters:
        mp4_path (str): Path to the audio/video file.
        target_fps (int): Desired audio sampling rate (Hz).

    Returns:
        tuple: (audio_array, sample_rate)
            audio_array: NumPy array of shape (n_samples, n_channels)
            sample_rate: Sampling rate in Hz
    """
    # Validate file path
    if not os.path.isfile(resource_path):
        raise FileNotFoundError(f"File not found: {resource_path}")
    if not resource_path.lower().endswith((".mp4", ".mp3", ".wav", ".flac")):
        raise ValueError("Input file must be an audio file or video.")

    try:
        # Load video
        clip = VideoFileClip(resource_path)

        if clip.audio is None:
            raise ValueError("No audio track found in the video.")

        # Convert audio to numpy array
        audio_array = clip.audio.to_soundarray(fps=target_fps)

        # Ensure it's a NumPy array
        audio_array = np.array(audio_array, dtype=np.float32)

        clip.close()
        return audio_array, target_fps

    except Exception as e:
        raise RuntimeError(f"Error extracting audio: {e}")

# Example usage in Jupyter:
audio_data, sample_rate = extract_audio_to_numpy(audio_file_path, target_fps=44100)
    
print(f"Audio shape: {audio_data.shape}")  # (samples, channels)
print(f"Sample rate: {sample_rate} Hz")
print(f"Print a small slice of the audio data:\n{audio_data[200000:200010]}")  # Print a small slice of the audio data

# Optional: Listen to audio in Jupyter
from IPython.display import Audio
Audio(audio_data.T, rate=sample_rate)  # Transpose for IPython Audio
