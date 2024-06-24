import librosa
import numpy as np
import matplotlib.pyplot as plt

'''
Ellis, Daniel PW. “Beat tracking by dynamic programming.” Journal of New Music Research 36.1 (2007): 51-60. http://labrosa.ee.columbia.edu/projects/beattrack/
'''

import librosa
import librosa.display
import matplotlib.pyplot as plt


def detect_beats_dp(audio_path, sr=44100, frame_length=2048, hop_length=512, *args, **kwargs):
    # 1. Preprocessing: Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # 2. Onset Detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)

    # 3. Tempo and Beat Tracking
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, trim=False, start_bpm=kwargs.get('start_bpm', 120))
    beats = beats
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    # print(f"Estimated Tempo: {tempo} beats per minute")

    # Plot beats on top of the onset envelope
    # plt.figure(figsize=(16, 5))
    # librosa.display.waveshow(y, sr=sr, alpha=0.5)
    # plt.vlines(beat_times, ymin=0, ymax=1, color='b', alpha=0.9, linestyle='-.', label='Beats')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Onset strength')
    # plt.title('Beats Overlaid on Onset Strength')
    # plt.legend()
    # plt.show()

    return beat_times


if __name__ == '__main__':
    audio_path = '../../audio/wlazl_kotek/wlazl_kotek.mp3'
    detect_beats_dp(audio_path, 44100, 2048, 512, 88)