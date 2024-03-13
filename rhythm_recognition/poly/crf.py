import madmom
import matplotlib.pyplot as plt
import numpy as np
import librosa


def detect_rhythm_with_madmom(audio_path):
    # 1. Preprocessing: Load the audio file using madmom
    proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path, start=0, stop=30)
    beats = proc(act)

    # Print detected beats
    print("Detected Beats (in seconds):")
    print(beats)

    sound = librosa.load(audio_path, sr=44100, duration=30)[0]

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(sound, sr=44100)

    # Plot the detected beats
    for beat in beats:
        plt.axvline(x=beat, color='r', linestyle='--', label='Beat' if beat == beats[0] else "")

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Detected Beats on Waveform')
    plt.show()


# Example usage
audio_path = '../../audio/midi_tracks/Canon_in_D.mp3'
detect_rhythm_with_madmom(audio_path)