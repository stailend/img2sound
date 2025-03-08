import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
from PIL import Image

# Load the image 
image_path = "image.png"  # Path to the image
image = Image.open(image_path).convert("L")  # Convert to grayscale
image_data = np.array(image)
image_bytes = image_data.flatten()

# Convert bytes to a sound wave 
audio_signal = (image_bytes - 128) / 128.0  # Normalize to range (-1,1)

# Save and play the sound
sample_rate = 44100  # Sampling rate
wav_filename = "encoded_audio.wav"
write(wav_filename, sample_rate, audio_signal.astype(np.float32))
