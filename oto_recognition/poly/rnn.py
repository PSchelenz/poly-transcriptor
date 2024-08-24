import madmom
from madmom.io.audio import load_ffmpeg_file, write_wave_file
import numpy as np
import matplotlib.pyplot as plt

'''
Sebastian Böck and Gerhard Widmer,
"Maximum Filter Vibrato Suppression for Onset Detection",
Proceedings of the 16th International Conference on Digital Audio
Effects (DAFx), 2013

https://www.dafx12.york.ac.uk/papers/dafx12_submission_4.pdf
'''

def detect_onsets_rnn(filename, sr=44100, hop_length=512, frame_length=2048, max_draw_note=88):
    # Load only a segment of the audio file
    # start_sec = 0
    # duration_sec = 30
    # audio, sr = load_ffmpeg_file(filename, start=start_sec, stop=start_sec + duration_sec)
    #
    # # Save the segment to a temporary file because RNNOnsetProcessor needs a file as input
    # temp_filename = 'temp_audio_segment.wav'
    # write_wave_file(audio, temp_filename, sr)

    # Initialize the pre-trained onset detection model
    # Dictionary for "Wlazł kotek"
    wlazl_kotek = {
        "pre_avg": 0.1,
        "post_avg": 0.1,
        "pre_max": 0.03,
        "post_max": 0.03,
        "combine": 0.03,
        "threshold": 0.2
    }

    # Dictionary for "Lot trzmiela"
    lot_trzmiela = {
        "pre_avg": 0.08,
        "post_avg": 0.08,
        "pre_max": 0.08,
        "post_max": 0.08,
        "combine": 0.07,
        "threshold": 0.02
    }

    # Dictionary for "Kołysanka"
    kolysanka = {
        "pre_avg": 0.08,
        "post_avg": 0.08,
        "pre_max": 0.1,
        "post_max": 0.1,
        "combine": 0.5,
        "threshold": 0.05
    }

    # Dictionary for "Kołysanka 2" (identical to "Kołysanka" based on provided information)
    kolysanka_2 = {
        "pre_avg": 0.4,
        "post_avg": 0.4,
        "pre_max": 0.1,
        "post_max": 0.1,
        "combine": 0.5,
        "threshold": 0.04
    }

    kolysanka_3 = {
        "pre_avg": 0.4,
        "post_avg": 0.4,
        "pre_max": 0.1,
        "post_max": 0.1,
        "combine": 0.5,
        "threshold": 0.05
    }

    dictionary = kolysanka_3

    proc = madmom.features.onsets.OnsetPeakPickingProcessor(
        pre_avg=dictionary["pre_avg"],
        post_avg=dictionary["post_avg"],
        pre_max=dictionary["pre_max"],
        post_max=dictionary["post_max"],
        combine=dictionary["combine"],
        threshold=dictionary["threshold"],
        fps=100
    )

    act = madmom.features.onsets.RNNOnsetProcessor(hop_size=hop_length)(filename)

    # Detect onsets
    onsets = proc(act)
    onsets = onsets * 100

    # Time vector for the audio segment
    # time = np.linspace(start_sec, start_sec + duration_sec, num=len(audio))

    # Plot the audio waveform of the segment and detected onsets
    # plt.figure(figsize=(14, 6))
    # plt.plot(time, audio, label='Audio Waveform (Segment)')
    # plt.vlines(onsets, ymin=0, ymax=max_draw_note, color='r', linestyle='--', label='Detected Onsets')
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Detected Onsets in Polyphonic Music (30-second Segment)')
    # plt.show()

    return onsets

if __name__ == '__main__':
    detected_onsets = detect_onsets_rnn('../../audio/midi_tracks/Canon_in_D.mp3', 512)