from definitions import ROOT_DIR
from midi_to_array import midi2array
from poly.dynamic_programming import detect_beats_dp
from poly.dbn import detect_beats_dbn
from poly.crf import detect_beats_crf
from rhythm_recognition.configurator import DATA_CONFIG, BEAT_DETECTOR_TO_FILENAME_MAPPERS, BEAT_DETECTOR_TO_HUMAN_READABLE

import matplotlib.pyplot as plt
import os
import numpy as np

def plot_midi_with_beats(midi_array, beats, data_scale, scaling_correction, correct_beats, tolerance):
    plt.figure(figsize=(14, 4))
    tp = 0
    fp = 0

    plt.plot(range(midi_array.shape[0]), np.multiply(np.where(midi_array > 0, 1, 0), range(1, 89)), marker='.',
             markersize=1, linestyle='')

    for beat in beats:
        beat = beat * data_scale - scaling_correction
        for cor_beat in correct_beats:
            error = abs(beat - cor_beat)

            if error < tolerance:
                color = 'g'
                tp += 1
                break

        else:
            color = 'r'
            fp += 1

        plt.axvline(x=beat, color=color, linestyle='--', label='Beats')

    # for beat in correct_beats:
    #     plt.axvline(x=beat, color='b', linestyle='--', label='Correct beats')

    fn = len(correct_beats) - tp

    f_score = f_measure(tp, fp, fn)

    plt.xlabel('Ramki czasowe')
    plt.ylabel('Dźwięki')
    plt.title(f'Bity ({CURR_DETECTOR_READABLE_NAME}, F-score = {f_score:.2f})')
    plt.ylim(0, 90)
    plt.savefig(os.path.join(DATA_CONFIG['save_to'], f'{CURR_TRACK_NAME}__{CURR_DETECTOR_NAME}.png'))
    plt.show()


def f_measure(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(tp, fp, fn)

    return 2 * precision * recall / (precision + recall)


def detect_beats(beats_detector_function, filename, sr=44100, hop_length=512, frame_length=2048, fps=100, start_bpm=120):
    beats = beats_detector_function(filename, sr, hop_length, frame_length, fps, start_bpm=start_bpm)

    return beats


if __name__ == '__main__':
    MIDI_RESOLUTION = DATA_CONFIG['midi_resolution']
    TRACK = DATA_CONFIG['tracks'][6]

    CURR_TRACK_NAME = TRACK['name']

    for beat_method in BEAT_DETECTOR_TO_FILENAME_MAPPERS.keys():
        CURR_DETECTOR_NAME = BEAT_DETECTOR_TO_FILENAME_MAPPERS[beat_method]
        CURR_DETECTOR_READABLE_NAME = BEAT_DETECTOR_TO_HUMAN_READABLE[beat_method]

        # All data about the track #
        source_dir = os.path.join(ROOT_DIR, TRACK['dir'])
        audio_track = os.path.join(source_dir, TRACK['audio'])
        midi_track = os.path.join(source_dir, TRACK['midi'])
        TRACK_SPEED = TRACK['qpm']  # quarters per minute
        # ------------------------ #

        # Initial calculation values #
        sample_rate = 44100
        hop_length = 512
        frame_length = 2048
        beats_detector = globals()[f'detect_beats_{beat_method}']
        scaling_correction = TRACK['scaling_correction']
        start_bpm = 240
        # -------------------------- #

        #    Calculate and plot     #
        beats = detect_beats(beats_detector,
                             audio_track,
                             sample_rate,
                             hop_length=hop_length,
                             frame_length=frame_length,
                             start_bpm=start_bpm)

        midi_array, _ = midi2array(midi_track)
        correct_beats = TRACK['correct_beats']

        # scale = midi resolution / time between quarter notes in seconds / frames per second
        if beat_method == 'crf' or beat_method == 'dbn':
            scale = MIDI_RESOLUTION / (60 / TRACK_SPEED) / 100  #CRF/DBN fps=100
        else:
            scale = MIDI_RESOLUTION / (60 / TRACK_SPEED) # DP

        # tolerance (number of frames) = 50ms * midi resolution / time between quarter notes in seconds
        tolerance = (TRACK['tolerance_s'] / 1000.) * (MIDI_RESOLUTION / (60 / TRACK_SPEED))

        plot_midi_with_beats(midi_array, beats, scale, scaling_correction, correct_beats, tolerance)
        # ------------------------- #
