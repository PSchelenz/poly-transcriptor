import madmom
import matplotlib.pyplot as plt
import numpy as np
import librosa

'''
[1]	Filip Korzeniowski, Sebastian Böck and Gerhard Widmer, “Probabilistic Extraction of Beat Positions from a Beat Activation Function”, Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR), 2014.
'''

def detect_beat_crf(audio_path, hop_length, max_draw_note=88):
    # 1. Preprocessing: Load the audio file using madmom
    proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=177)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path, start=0, stop=30)
    beats = proc(act)
    beats = beats * 177 / 4
    #
    # # Print detected beats
    # print("Detected Beats (in seconds):")
    # print(beats)
    #
    # sound = librosa.load(audio_path, sr=44100, duration=30)[0]

    # Plot the waveform
    # plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(sound, sr=44100)

    # Plot the detected beats
    for beat in beats:
        plt.axvline(x=beat, ymin=0, ymax=max_draw_note, color='b', linestyle='-.', label='Beat' if beat == beats[0] else "")

    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Detected Beats on Waveform')
    # plt.show()

    return beats

if __name__ == '__main__':
    # Example usage
    audio_path = '../../audio/midi_tracks/Canon_in_D.mp3'
    detect_beat_crf(audio_path)