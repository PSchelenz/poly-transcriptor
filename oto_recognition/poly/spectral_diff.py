import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

'''
Chris Duxbury, Mark Sandler and Matthew Davis, “A hybrid approach to musical note onset detection”, Proceedings of the 5th International Conference on Digital Audio Effects (DAFx), 2002.
'''

def onset_detection_spectral_diff_and_plot(audio_file, sr=44100, hop_length=512, frame_length=1024):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=sr, duration=30)

    # Calculate the Short-Time Fourier Transform (STFT)
    D = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    # Calculate spectral difference
    spectral_diff = np.abs(np.diff(D, axis=1))

    # Sum the spectral difference over frequency bins to get a 'spectral difference function'
    diff_sum = np.sum(spectral_diff, axis=0)

    # Detect onsets by finding peaks in the spectral difference function
    onset_frames = librosa.util.peak_pick(diff_sum, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.1, wait=0)

    # Convert frames to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Plotting
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.vlines(onset_times, -1, 1, color='r', linestyle='--', label='Onsets')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform and Detected Onsets')
    plt.legend()
    plt.show()

    return onset_times


# Example usage
audio_file = '../../audio/midi_tracks/Canon_in_D.mp3'
onset_times = onset_detection_spectral_diff_and_plot(audio_file)