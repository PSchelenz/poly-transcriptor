import madmom
import matplotlib.pyplot as plt
from madmom.audio.filters import LogarithmicFilterbank
from madmom.io.audio import load_ffmpeg_file, write_wave_file
import numpy as np

'''
Sebastian Böck and Gerhard Widmer,
"Maximum Filter Vibrato Suppression for Onset Detection",
Proceedings of the 16th International Conference on Digital Audio
Effects (DAFx), 2013

https://www.dafx12.york.ac.uk/papers/dafx12_submission_4.pdf
'''

def onset_detection_phase_deviation(filename, hop_length, max_draw_note=88):
    # Load only a segment of the audio file
    # audio, sr = load_ffmpeg_file(filename, start=0, stop=30, sample_rate=44100)

    # Save the segment to a temporary file because RNNOnsetProcessor needs a file as input
    # temp_filename = 'temp_audio_segment.wav'
    # write_wave_file(audio, temp_filename, sr)

    # Initialize the pre-trained onset detection model
    proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=177, pre_max=0.25, post_max=0.25, pre_avg=0.25, post_avg=0.25)
    act = madmom.features.onsets.SpectralOnsetProcessor('phase_deviation', fps=177)(filename, start=0, stop=30)

    # Detect onsets
    onsets = proc(act)
    onsets = onsets * 177 / 7.075

    # Time vector for the audio segment
    # time = np.linspace(0, 30, num=len(audio))

    # Plot the audio waveform of the segment and detected onsets
    # plt.figure(figsize=(14,4))
    # plt.plot(time, audio, label='Audio Waveform (Segment)')
    plt.vlines(onsets, ymin=0, ymax=max_draw_note, color='r', linestyle='--', label='Detected Onsets')
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Detected Onsets in Polyphonic Music (30-second Segment)')
    # plt.show()

    return onsets

if __name__ == '__main__':
    detected_onsets = onset_detection_phase_deviation('../../audio/midi_tracks/Canon_in_D.mp3', 512)