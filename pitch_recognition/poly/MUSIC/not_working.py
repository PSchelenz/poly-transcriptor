import numpy as np
import librosa
import matplotlib.pyplot as plt

def load_audio(filename, sr, duration):
    # Load an audio file, returning the signal and sampling rate
    signal, sr = librosa.load(filename, sr=sr, duration=duration, mono=True)
    return signal, sr

def frame_signal(signal, frame_size, hop_size):
    # Frame the signal with a window function
    frames = librosa.util.frame(signal, frame_length=frame_size, hop_length=hop_size).astype(np.float64)
    window = np.hanning(frame_size)  # Using Hann window for smoothing
    windowed_frames = frames * window[:, None]
    return windowed_frames

def fft_frames(frames):
    # Apply FFT to each column (frame)
    return np.fft.rfft(frames, axis=0)

def estimate_covariance_matrix(frames):
    # Estimate the covariance matrix from frames
    # Assuming frames are already in the frequency domain
    num_frames = frames.shape[1]
    covariance_matrix = np.dot(frames, frames.conj().T) / num_frames
    return covariance_matrix

def perform_eigen_decomposition(covariance_matrix):
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvectors by eigenvalues in ascending order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

# MUSIC pseudo-spectrum calculation
def steering_vector(freq, fs):
    num_elements = eigenvectors.shape[0]
    return np.exp(-2j * np.pi * freq * np.arange(num_elements) / fs)

def compute_music_pseudospectrum(eigenvectors, num_signal_components, freqs, fs):
    # Noise subspace (eigenvectors corresponding to smallest eigenvalues)
    noise_subspace = eigenvectors[:, :eigenvectors.shape[1]-num_signal_components]

    spectrum = np.array([1.0 / np.abs(steering_vector(freq, fs) @ noise_subspace @ noise_subspace.conj().T @ steering_vector(freq, fs))**2
                         for freq in freqs])

    return spectrum

def find_peaks(spectrum, freqs, threshold=0.1):
    # Detect peaks in the MUSIC spectrum
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spectrum, height=threshold)
    peak_freqs = freqs[peaks]
    peak_values = spectrum[peaks]
    return peak_freqs, peak_values

def freq_to_note(freq):
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    if freq == 0:
        return "None"  # No frequency found
    h = round(12 * np.log2(freq / C0))
    octave = h // 12
    n = h % 12
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H']
    return f"{notes[n]}{octave}"

def _plot(signal):
    plt.figure()
    plt.plot(signal)
    plt.title('_plot')
    plt.show()

# Example usage
filename = '../../../audio/midi_tracks/Canon_in_D.mp3'  # Adjust this path to your audio file
signal, sr = load_audio(filename, 44100, 1)
frames = frame_signal(signal, frame_size=2048, hop_size=512)  # Example frame and hop sizes

# Define frequency range for analysis
frequencies = np.linspace(20, 2000, 3000)  # Avoid too low frequencies which are not musical notes

# Process each frame
detected_notes_per_frame = []
for i, frame in enumerate(frames.T[10:]):
    print(f"Frame {i+1}.")
    _plot(frame)

    frequency_domain_frame = np.fft.rfft(frame)

    _plot(frequency_domain_frame)

    covariance_matrix = estimate_covariance_matrix(frequency_domain_frame.reshape(-1, 1))
    eigenvalues, eigenvectors = perform_eigen_decomposition(covariance_matrix)
    num_sources = 1  # Assuming two sources for simplicity, adjust based on your needs
    music_pseudo_spectrum = compute_music_pseudospectrum(eigenvectors, num_sources, frequencies, sr)

    plt.figure()
    plt.plot(frequencies, music_pseudo_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Pseudo Spectrum')
    plt.title('MUSIC Pseudo Spectrum')
    plt.show()

    peak_freqs, _ = find_peaks(music_pseudo_spectrum, frequencies, threshold=60000)
    # notes = [freq_to_note(freq) for freq in peak_freqs]
    print(f"Frame {i}: Notes Detected: {peak_freqs}")
    # detected_notes_per_frame.append(notes)

# Print or further process the detected notes per frame
for i, notes in enumerate(detected_notes_per_frame):
    print(f"Frame {i}: Notes Detected: {notes}")