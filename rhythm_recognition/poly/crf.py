import madmom
import matplotlib.pyplot as plt
import numpy as np
import librosa
from midi_to_array import midi2array

'''
[1]	Filip Korzeniowski, Sebastian Böck and Gerhard Widmer, “Probabilistic Extraction of Beat Positions from a Beat Activation Function”, Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR), 2014.
'''

def detect_beats_crf(audio_path, sr, frame_length, hop_length=512, fps=100):
    # 1. Preprocessing: Load the audio file using madmom
    proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=fps)
    act = madmom.features.beats.RNNBeatProcessor(fps=fps)(audio_path)
    beats = proc(act)
    beats = beats * fps
    #
    # # Print detected beats
    # print("Detected Beats (in seconds):")
    # print(beats)
    #
    # sound = librosa.load(audio_path, sr=44100)[0]

    # Plot the waveform
    # plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(sound, sr=44100)

    # Plot the detected beats
    # for beat in beats:
    #     plt.axvline(x=beat, ymin=0, ymax=88, color='b', linestyle='-.', label='Beat' if beat == beats[0] else "")

    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Detected Beats on Waveform')
    # plt.show()

    return beats

if __name__ == '__main__':
    # Example usage
    audio_path = '../../audio/lot_trzmiela/lot-trzmiela.mp3'

    detect_beats_crf(audio_path, 44100, 2048, 512, 100)