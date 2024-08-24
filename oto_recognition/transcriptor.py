from poly.hfc import detect_onsets_hfc
from poly.phase_deviation import detect_onsets_pd
from poly.spectral_flux import detect_onsets_sf
from poly.spectral_diff import detect_onsets_sd
from poly.rnn import detect_onsets_rnn
from definitions import ROOT_DIR
from oto_recognition.configurator import DATA_CONFIG, ONSET_DETECTOR_TO_FILENAME_MAPPERS, \
    ONSET_DETECTOR_TO_HUMAN_READABLE
from midi_to_array import midi2array

import matplotlib.pyplot as plt
import os
import numpy as np

TRACK_NUM = 4
CORRECT_FIX = 0
PREDICTED_FIX = 0


def plot_midi_with_onsets(midi_array, onsets, data_scale, scaling_correction, correct_onsets, tolerance):
    global CURR_TRACK_NAME, CURR_DETECTOR_NAME, CURR_DETECTOR_READABLE_NAME
    tp = 0
    fp = 0

    if CORRECT_FIX != 0:
        correct_onsets = correct_onsets[:CORRECT_FIX]

    # todo: for kolysanka
    correct_onsets = np.append(correct_onsets, [9360])
    correct_onsets = np.delete(correct_onsets, -7)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(range(midi_array.shape[0]), np.multiply(np.where(midi_array > 0, 1, 0), range(1, 89)), marker='.',
                markersize=1, linestyle='')
    axs[0].set_ylim(1, 90)
    axs[0].set_title('MIDI')
    axs[0].set_ylabel('Numery dźwięków')

    for onset in correct_onsets:
        axs[0].axvline(x=onset, color='blue', linestyle='--', linewidth=1, label='Onsets')

    axs[1].plot(range(midi_array.shape[0]), np.multiply(np.where(midi_array > 0, 1, 0), range(1, 89)), marker='.',
                markersize=1, linestyle='')
    axs[1].set_ylim(1, 90)
    axs[1].set_title('Predykcja')
    axs[1].set_ylabel('Numery dźwięków')

    for onset in onsets:
        onset = onset * data_scale - scaling_correction
        axs[1].axvline(x=onset, color='blue', linestyle='--', linewidth=1, label='Onsets')

    axs[2].plot(range(midi_array.shape[0]), np.multiply(np.where(midi_array > 0, 1, 0), range(1, 89)), marker='.',
                markersize=1, linestyle='')
    axs[2].set_ylim(1, 90)

    for onset in onsets:
        onset = onset * data_scale - scaling_correction

        for cor_onset in correct_onsets:
            error = abs(onset - cor_onset)

            if error < tolerance:
                color = 'g'
                tp += 1
                break
        else:
            color = 'r'
            fp += 1

        axs[2].axvline(x=onset, color=color, linestyle='--', linewidth=1, label='Onsets')

    fn = len(correct_onsets) - tp  # -1

    metrics = calculate_metrics(tp, fp, fn)

    axs[2].set_title(f'Trafność ({CURR_DETECTOR_READABLE_NAME}, F-score = {metrics["F-score"]:.2f})')
    axs[2].set_ylabel('Numery dźwięków')

    fig.tight_layout(h_pad=2.0)
    plt.subplots_adjust(bottom=.05, top=.95)
    plt.xlabel('Ramki czasowe')
    plt.savefig(os.path.join(DATA_CONFIG['save_to'], f'{CURR_TRACK_NAME}__{CURR_DETECTOR_NAME}.png'))
    plt.show()

    print(metrics)


def f_measure(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(tp, fp, fn)

    return 2 * precision * recall / (precision + recall)


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": precision,
        "Recall": recall,
        "F-score": f_score
    }


def detect_onsets(onsets_detector_function, filename, sr=44100, hop_length=512, frame_length=2048, max_draw_note=88):
    onsets = onsets_detector_function(filename, sr, hop_length, frame_length, max_draw_note)

    return onsets


if __name__ == '__main__':
    MIDI_RESOLUTION = DATA_CONFIG['midi_resolution']
    TRACK = DATA_CONFIG['tracks'][TRACK_NUM]

    CURR_TRACK_NAME = TRACK['name']

    for onset_method in ONSET_DETECTOR_TO_FILENAME_MAPPERS.keys():
        CURR_DETECTOR_NAME = ONSET_DETECTOR_TO_FILENAME_MAPPERS[onset_method]
        CURR_DETECTOR_READABLE_NAME = ONSET_DETECTOR_TO_HUMAN_READABLE[onset_method]

        # All data about the track #
        source_dir = os.path.join(ROOT_DIR, TRACK['dir'])
        audio_track = os.path.join(source_dir, TRACK['audio'])
        midi_track = os.path.join(source_dir, TRACK['midi'])
        TRACK_SPEED = TRACK['qpm']  # quarters per minute
        # ------------------------ #

        # Initial calculation values #
        sample_rate = 44100
        hop_length = 256
        frame_length = 4096
        onsets_detector = globals()[f'detect_onsets_{onset_method}']
        scaling_correction = TRACK['scaling_correction']

        # -------------------------- #

        #    Calculate and plot     #
        onsets = detect_onsets(onsets_detector,
                               audio_track,
                               sample_rate,
                               hop_length=hop_length,
                               frame_length=frame_length)

        if PREDICTED_FIX != 0:
            onsets = onsets[:PREDICTED_FIX]

        midi_array, note_change_times = midi2array(midi_track)

        if 0 not in note_change_times:
            note_change_times = np.insert(note_change_times, 0, 0)
            note_change_times = note_change_times[:-1]

        # scale = midi resolution / time between quarter notes in seconds / frames per second
        scale = MIDI_RESOLUTION / (60 / TRACK_SPEED) / (sample_rate / hop_length)

        # tolerance (number of frames) = 50ms * midi resolution / time between quarter notes in seconds
        tolerance = (TRACK['tolerance_s'] / 1000.) * (MIDI_RESOLUTION / (60 / TRACK_SPEED))

        plot_midi_with_onsets(midi_array, onsets, scale, scaling_correction, note_change_times, tolerance)
        # ------------------------- #
