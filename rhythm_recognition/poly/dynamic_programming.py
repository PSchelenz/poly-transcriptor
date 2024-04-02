import librosa
import numpy as np
import matplotlib.pyplot as plt

'''
Ellis, Daniel PW. “Beat tracking by dynamic programming.” Journal of New Music Research 36.1 (2007): 51-60. http://labrosa.ee.columbia.edu/projects/beattrack/
'''

import librosa
import librosa.display
import matplotlib.pyplot as plt


def dynamic_programming_beat_recognition(audio_path, hop_length=512, max_draw_note=88):
    # 1. Preprocessing: Load the audio file
    y, sr = librosa.load(audio_path, duration=30)

    # 2. Onset Detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 3. Tempo and Beat Tracking
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beats = beats / 3.45
    # beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    # print(f"Estimated Tempo: {tempo} beats per minute")

    # Plot beats on top of the onset envelope
    # plt.figure(figsize=(16, 5))
    # librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.vlines(beats, ymin=0, ymax=max_draw_note, color='b', alpha=0.9, linestyle='-.', label='Beats')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Onset strength')
    # plt.title('Beats Overlaid on Onset Strength')
    # plt.legend()
    # plt.show()

    return beats


if __name__ == '__main__':
    audio_path = '../../audio/midi_tracks/Canon_in_D.mp3'
    dynamic_programming_beat_recognition(audio_path)