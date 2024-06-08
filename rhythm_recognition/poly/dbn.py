import madmom
import matplotlib.pyplot as plt
import numpy as np
import librosa

'''
[1]	Sebastian Böck, Florian Krebs and Gerhard Widmer, “A Multi-Model Approach to Beat Tracking Considering Heterogeneous Music Styles”, Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR), 2014.
[2]	Florian Krebs, Sebastian Böck and Gerhard Widmer, “An Efficient State Space Model for Joint Tempo and Meter Tracking”, Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR), 2015.
'''

def detect_beats_dbn(audio_path, sr, frame_length, hop_length=512, fps=100):
    # 1. Preprocessing: Load the audio file using madmom
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=fps)
    act = madmom.features.beats.RNNBeatProcessor(fps=fps)(audio_path)
    beats = proc(act)
    beats = beats * fps
    #
    # # Print detected beats
    # print("Detected Beats (in seconds):")
    # print(beats)
    #
    # sound = librosa.load(audio_path, sr=44100, duration=30)[0]
    #
    # # Plot the waveform
    # plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(sound, sr=44100)

    # Plot the detected beats
    # for beat in beats:
    #     plt.axvline(x=beat, ymin=0, ymax=max_draw_note, color='b', linestyle='-.', label='Beat' if beat == beats[0] else "")

    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Detected Beats on Waveform')
    # plt.show()

    return beats


if __name__ == '__main__':
    audio_path = '../../audio/midi_tracks/Canon_in_D.mp3'
    detect_beats_dbn(audio_path)