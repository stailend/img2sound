import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file 
wav_filename = "encoded_audio.wav"
sample_rate = 44100  # Must match the encoding
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Define control sequences
START_SEQUENCE = np.array([255, 0, 255, 0, 255, 0, 255, 0], dtype=np.uint8)
ROW_SEPARATOR = np.array([127, 127, 127, 127], dtype=np.uint8)
END_SEQUENCE = np.array([0, 255, 0, 255, 0, 255, 0, 255], dtype=np.uint8)

# Load recorded audio
wav_filename = "encoded_audio.wav"
sample_rate = 44100
y, sr = librosa.load(wav_filename, sr=sample_rate)

# Apply low-pass filter to remove noise
def butter_lowpass_filter(data, cutoff=5000, fs=44100, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

y_filtered = butter_lowpass_filter(y, cutoff=4000, fs=44100, order=6)

# Convert back to byte values
image_bytes_recovered = ((y_filtered + 1) * 128).astype(np.uint8)

# Function to find the closest match using cross-correlation
def find_sequence_cross_corr(data, sequence, window_size=500, threshold=0.8):
    seq_len = len(sequence)
    max_corr = -1
    best_idx = None

    for i in range(len(data) - seq_len):
        segment = data[i:i+seq_len]
        corr = np.correlate(segment - np.mean(segment), sequence - np.mean(sequence))[0]
        norm_corr = corr / (np.linalg.norm(segment) * np.linalg.norm(sequence) + 1e-6)  # Normalize
        
        if norm_corr > max_corr:
            max_corr = norm_corr
            best_idx = i
            
    return best_idx if max_corr > threshold else None

# Find start and end markers with correlation
start_idx = find_sequence_cross_corr(image_bytes_recovered, START_SEQUENCE)
end_idx = find_sequence_cross_corr(image_bytes_recovered, END_SEQUENCE)

if start_idx is None or end_idx is None:
    print("❌ Error: Start or end sequence not found. Trying adaptive threshold...")
    start_idx = find_sequence_cross_corr(image_bytes_recovered, START_SEQUENCE, threshold=0.7)
    end_idx = find_sequence_cross_corr(image_bytes_recovered, END_SEQUENCE, threshold=0.7)

if start_idx is None or end_idx is None:
    print("❌ Start or end sequence still not found. Image cannot be reconstructed.")
    exit()

print(f"🔍 Found START at {start_idx}, END at {end_idx}")

# Extract image bytes
image_bytes_recovered = image_bytes_recovered[start_idx + len(START_SEQUENCE): end_idx]

# Remove row separators
rows = []
current_row = []
for i, byte in enumerate(image_bytes_recovered):
    if len(current_row) >= 4 and find_sequence_cross_corr(np.array(current_row[-4:]), ROW_SEPARATOR, threshold=0.7) is not None:
        checksum = current_row[-5] if len(current_row) > 5 else 0
        row_data = np.array(current_row[:-5], dtype=np.uint8)
        
        if np.sum(row_data) % 256 == checksum:
            rows.append(row_data)
        else:
            print(f"⚠️ Checksum mismatch at row {len(rows)}")
        
        current_row = []
    else:
        current_row.append(byte)

# If no valid rows were found, exit
if len(rows) == 0:
    print("❌ No valid rows found. Image cannot be reconstructed.")
    exit()

# Convert to NumPy array and reshape
image_recovered = np.vstack(rows)

# Display the recovered image
plt.figure(figsize=(6,6))
plt.imshow(image_recovered, cmap="gray")
plt.title("Recovered Image")
plt.axis("off")
plt.show()
print("🎵 Loading audio file...")
y, sr = librosa.load(wav_filename, sr=sample_rate)

# Remove silence from the signal 
y_trimmed, _ = librosa.effects.trim(y, top_db=20)

# Convert audio back to 0-255 grayscale values 
image_bytes_recovered = ((y_trimmed + 1) * 128).astype(np.uint8)

# Determine image size 
size = int(np.sqrt(len(image_bytes_recovered)))
image_recovered = image_bytes_recovered[:size * size].reshape((size, size))

# Display the recovered image 
plt.figure(figsize=(6,6))
plt.imshow(image_recovered, cmap="gray")
plt.title("Recovered Image from .wav")
plt.axis("off")
plt.show()