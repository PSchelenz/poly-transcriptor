from poly.hfc import detect_onsets_hfc
from poly.phase_deviation import detect_onsets_pd
from poly.spectral_flux import detect_onsets_sf
from poly.spectral_diff import detect_onsets_sd
from poly.rnn import detect_onsets_rnn
from definitions import ROOT_DIR
from midi_to_array import midi2array

import matplotlib.pyplot as plt
import os
import numpy as np

MIDI_RESOLUTION = 480

def plot_midi_with_onsets(midi_array, onsets, data_scale, scaling_correction, correct_onsets, tolerance):
    plt.figure(figsize=(14, 4))

    plt.plot(range(midi_array.shape[0]), np.multiply(np.where(midi_array > 0, 1, 0), range(1, 89)), marker='.',
             markersize=1, linestyle='')

    for onset in onsets:
        onset = onset * data_scale - scaling_correction

        for cor_onset in correct_onsets:
            if abs(onset - cor_onset) < tolerance:
                color = 'g'
                break
        else:
            color = 'r'

        plt.axvline(x=onset, color=color, linestyle='--', label='Onsets')

    plt.xlabel('Frame')
    plt.ylabel('Notes')
    plt.title('Detected Onsets in Polyphonic Music')
    plt.ylim(0, 90)
    plt.show()


def detect_onsets(onsets_detector_function, filename, sr=44100, hop_length=512, frame_length=2048, max_draw_note=88):
    onsets = onsets_detector_function(filename, sr, hop_length, frame_length, max_draw_note)

    return onsets


if __name__ == '__main__':
    # All data about the track #
    source_dir = os.path.join(ROOT_DIR, 'audio/wlazl_kotek')
    audio_track = os.path.join(source_dir, 'wlazl_kotek.mp3')
    midi_track = os.path.join(source_dir, 'wlazl_kotek.mid')
    TRACK_SPEED = 100  # quarters per minute
    # ------------------------ #

    # Initial calculation values #
    sample_rate = 44100
    hop_length = 512
    frame_length = 2048
    max_draw_note = 88
    onsets_detector = detect_onsets_sd
    scaling_correction = 50
    # -------------------------- #

    #    Calculate and plot     #
    onsets = detect_onsets(onsets_detector,
                           audio_track,
                           sample_rate,
                           hop_length=hop_length,
                           frame_length=frame_length,
                           max_draw_note=max_draw_note)

    midi_array, note_change_times = midi2array(midi_track)

    if 0 not in note_change_times:
        note_change_times = [0] + note_change_times

    # scale = midi resolution / time between quarter notes in seconds / frames per second
    scale = MIDI_RESOLUTION / (60 / TRACK_SPEED) / (sample_rate / hop_length)

    # tolerance (number of frames) = 50ms * midi resolution / time between quarter notes in seconds
    tolerance = 0.05 * (MIDI_RESOLUTION / (60 / TRACK_SPEED))

    plot_midi_with_onsets(midi_array, onsets, scale, scaling_correction, note_change_times, tolerance)
    # ------------------------- #
