import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

'''
Paul Masri, “Computer Modeling of Sound for Transformation and Synthesis of Musical Signals”, PhD thesis, University of Bristol, 1996.
'''

def detect_onsets_sf(filename, sr=44100, hop_length=512, frame_length=2048, max_draw_note=88):
    # Load the audio file
    y, sr = librosa.load(filename, sr=sr)

    # Compute the short-time Fourier transform (STFT)
    D = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    S = librosa.amplitude_to_db(D, ref=np.max)

    # Compute spectral flux
    onset_env = librosa.onset.onset_strength(S = S, hop_length=hop_length, sr=sr)

    # Detect onsets
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    # Convert frame indices to time
    # onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

    # Plotting
    # plt.figure(figsize=(12, 8))

    # Plot the waveform
    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(y, sr=sr, alpha=0.6)
    # plt.vlines(onset_times, ymin=-1, ymax=1, color='r', linestyle='--', label='Onsets')
    # plt.legend()
    # plt.title('Waveform with Onsets')

    # Plot the spectrogram
    # plt.subplot(2, 1, 2)
    # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    #                          y_axis='log', x_axis='time')


    # plt.vlines(onsets / 3.45, ymin=0, ymax=max_draw_note, color='r', linestyle='--', label='Onsets')


    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectrogram with Onsets')
    # plt.tight_layout()

    # return onset_times
    return onsets

if __name__ == '__main__':
    # Replace 'path/to/your/music/file' with the actual path to your audio file
    onset_times = detect_onsets_sf('../../audio/midi_tracks/Canon_in_D.mp3', 44100, 512)
    plt.show()