import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

'''
Paul Masri, “Computer Modeling of Sound for Transformation and Synthesis of Musical Signals”, PhD thesis, University of Bristol, 1996.
'''

def detect_onsets(filename):
    # Load the audio file
    y, sr = librosa.load(filename, duration=30)

    # Compute the short-time Fourier transform (STFT)
    D = np.abs(librosa.stft(y))

    # Compute spectral flux
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Detect onsets
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    # Convert frame indices to time
    onset_times = librosa.frames_to_time(onsets, sr=sr)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot the waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.vlines(onset_times, ymin=-1, ymax=1, color='r', linestyle='--', label='Onsets')
    plt.legend()
    plt.title('Waveform with Onsets')

    # Plot the spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', x_axis='time')
    plt.vlines(onset_times, ymin=0, ymax=D.shape[0], color='r', linestyle='--', label='Onsets')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with Onsets')
    plt.tight_layout()

    plt.show()

    return onset_times

# Replace 'path/to/your/music/file' with the actual path to your audio file
onset_times = detect_onsets('../../audio/midi_tracks/Canon_in_D.mp3')