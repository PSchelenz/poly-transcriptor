import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


def load_audio(filename, sr, duration):
    x, fs = librosa.load(filename, sr=sr, duration=duration, mono=True)  # Load audio file
    return x, fs


def estimate_covariance_matrix(x, M):
    N = len(x)
    num_segments = N - M + 1
    X = np.array([x[n:n + M] for n in range(num_segments)]).T
    R = np.cov(X)
    return R


def compute_subspaces(R, num_sources):
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(R)
    # Sort eigenvalues in descending order and reorder eigenvectors accordingly
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract the noise subspace (eigenvectors corresponding to smallest eigenvalues)
    noise_subspace = eigenvectors[:, num_sources:]  # Assuming 'num_sources' is known
    return noise_subspace


def music_pseudo_spectrum(noise_subspace, freq_grid, fs, M):
    # Define the steering vector for each frequency
    spectrum = np.zeros_like(freq_grid)
    for i, freq in enumerate(freq_grid):
        # Steering vector for the frequency
        steering_vector = np.exp(-2j * np.pi * freq * np.arange(M) / fs)
        # MUSIC pseudo-spectrum calculation
        # spectrum[i] = 1 / np.abs(
        #     steering_vector.conj().dot(noise_subspace).dot(noise_subspace.conj().T).dot(steering_vector))

        projection = steering_vector.conj().dot(noise_subspace)
        spectrum[i] = 1 / np.linalg.norm(projection) ** 2
    return spectrum


def find_peaks(spectrum, freq_grid, num_peaks):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.1)  # Threshold can be adjusted
    peak_freqs = freq_grid[peaks]
    # Select top 'num_peaks' based on highest values in the spectrum
    top_indices = np.argsort(spectrum[peaks])[-num_peaks:]
    return peak_freqs[top_indices]


def freq_to_note(freqs):

    A4 = 440
    C0 = A4 * pow(2, -4.75)

    printer = []

    for freq in freqs:
        h = round(12 * np.log2(freq / C0))
        octave = h // 12
        n = h % 12
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H']

        if f"{notes[n]}{octave}" in printer:
            continue
        else:
            printer.append(f"{notes[n]}{octave}")

    print(printer)

def _plot(signal):
    plt.figure()
    plt.plot(signal)
    plt.title('_plot')
    plt.show()

# Assuming you have an audio file named 'example.wav'
audio_file = '../../../audio/midi_tracks/Canon_in_D.mp3'
x, fs = load_audio(audio_file, 22050, 30)

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(x)
plt.title('Audio Waveform')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()

M = int(fs / 24)  # Modify as needed based on the lowest frequency
R = estimate_covariance_matrix(x, M)

num_sources = 1  # This needs to be estimated or predefined
noise_subspace = compute_subspaces(R, num_sources)

# Define segment length and hop size
segment_length = int(fs * 0.1)  # e.g., 100 ms segments
hop_length = int(segment_length * 0.5)  # 50% overlap

# Process each segment
for start in range(0, len(x) - segment_length, hop_length):
    segment = x[start:start + segment_length]
    _plot(segment)

    # Windowing
    windowed_segment = segment * np.hanning(segment_length)
    # _plot(windowed_segment)

    # Estimate covariance matrix
    R = estimate_covariance_matrix(windowed_segment, M)

    # Assuming you know num_sources for this segment
    noise_subspace = compute_subspaces(R, num_sources)
    # _plot(noise_subspace[:, 0])

    # Define a relevant frequency grid
    freq_grid = np.linspace(0, 4000, 1000)

    # Compute MUSIC pseudo-spectrum for the segment
    spectrum = music_pseudo_spectrum(noise_subspace, freq_grid, fs, M)

    # Plot the pseudo-spectrum
    plt.figure()
    plt.plot(freq_grid, spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Pseudo Spectrum')
    plt.title('MUSIC Pseudo Spectrum')
    plt.show()

    # Find peaks (frequencies)
    estimated_freqs = find_peaks(spectrum, freq_grid, num_sources)

    # Map frequencies to musical notes here (not shown)

    # Optionally store/plot results per segment
    freq_to_note(estimated_freqs)
