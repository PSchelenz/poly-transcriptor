import librosa
import numpy as np
import matplotlib.pyplot as plt

'''
Ellis, Daniel PW. “Beat tracking by dynamic programming.” Journal of New Music Research 36.1 (2007): 51-60. http://labrosa.ee.columbia.edu/projects/beattrack/
'''

import librosa
import librosa.display
import matplotlib.pyplot as plt


def detect_rhythm(audio_path):
    # 1. Preprocessing: Load the audio file
    y, sr = librosa.load(audio_path, duration=30)

    # 2. Onset Detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # 3. Tempo and Beat Tracking
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    print(f"Estimated Tempo: {tempo} beats per minute")

    # Plot beats on top of the onset envelope
    plt.figure(figsize=(16, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.vlines(beat_times, -1, y.max(), color='r', alpha=0.9, linestyle='--', label='Beats')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset strength')
    plt.title('Beats Overlaid on Onset Strength')
    plt.legend()
    plt.show()


# Example usage
audio_path = '../../audio/midi_tracks/Canon_in_D.mp3'
detect_rhythm(audio_path)