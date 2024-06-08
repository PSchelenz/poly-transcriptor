from definitions import ROOT_DIR
from midi_to_array import midi2array
from poly.dynamic_programming import detect_beats_dp
from poly.dbn import detect_beats_dbn
from poly.crf import detect_beats_crf

import matplotlib.pyplot as plt
import os
import numpy as np

MIDI_RESOLUTION = 480

tracks_bpm = {
    'wlazl_kotek': 100,
    'lot-trzmiela': 140,
    'blue-rondo': 126,
    'ipanema': 120,
    'juice': 110,
    'kolysanka': 60,
    'a_kiedy': 146,
}


def plot_midi_with_beats(midi_array, beats, data_scale, scaling_correction):
    plt.figure(figsize=(14, 4))

    plt.plot(range(midi_array.shape[0]), np.multiply(np.where(midi_array > 0, 1, 0), range(1, 89)), marker='.',
             markersize=1, linestyle='')

    for beat in beats:
        beat = beat * data_scale - scaling_correction
        plt.axvline(x=beat, color='b', linestyle='--', label='Beats')

    plt.xlabel('Frame')
    plt.ylabel('Notes')
    plt.title('Detected Beats in Polyphonic Music')
    plt.ylim(0, 90)
    plt.show()


def detect_beats(beats_detector_function, filename, sr=44100, hop_length=512, frame_length=2048, fps=20):
    beats = beats_detector_function(filename, sr, hop_length, frame_length, fps)

    return beats


if __name__ == '__main__':
    # All data about the track #
    source_dir = os.path.join(ROOT_DIR, 'audio/wlazl_kotek')
    audio_track = os.path.join(source_dir, 'wlazl_kotek.mp3')
    midi_track = os.path.join(source_dir, 'wlazl_kotek.mid')
    TRACK_SPEED = tracks_bpm['wlazl_kotek']  # quarters per minute
    # ------------------------ #

    # Initial calculation values #
    sample_rate = 44100
    hop_length = 512
    frame_length = 2048
    fps = 100
    scaling_correction = 60
    beats_detector = detect_beats_dp
    # -------------------------- #

    #    Calculate and plot     #
    beats = detect_beats(beats_detector,
                         audio_track,
                         sample_rate,
                         hop_length=hop_length,
                         frame_length=frame_length,
                         fps=fps)

    midi_array, _ = midi2array(midi_track)

    # scale = midi resolution / time between quarter notes in seconds / frames per second
    # scale = MIDI_RESOLUTION / (60 / TRACK_SPEED) / fps  #CRF/DBN
    scale = MIDI_RESOLUTION / (60 / TRACK_SPEED) # DP

    # tolerance (number of frames) = 50ms * midi resolution / time between quarter notes in seconds
    tolerance = 0.05 * (MIDI_RESOLUTION / (60 / TRACK_SPEED))

    plot_midi_with_beats(midi_array, beats, scale, scaling_correction)
    # ------------------------- #
