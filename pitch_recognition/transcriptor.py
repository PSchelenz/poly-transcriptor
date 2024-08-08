from definitions import ROOT_DIR
from midi_to_array import midi2array
from poly.cbpdn.cbpdn_algo_v2 import detect_pitch_cbpdn
from poly.pi2.algo import detect_pitch_pi2
# from poly.karjalainen_tolonen.algo import detect_pitch_kt
from poly.karjalainen_tolonen.algo_v2 import detect_pitch_kt
from poly.si_plca.si_plca_algo import detect_pitch_siplca
from pitch_recognition.configurator import DATA_CONFIG, PITCH_DETECTOR_TO_FILENAME_MAPPERS, \
    PITCH_DETECTOR_TO_HUMAN_READABLE

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np
import math
from scipy import ndimage


def pad_uneven_array(uneven_array, target_shape):
    # Create a zero-filled array with the target shape
    padded_array = np.zeros(target_shape, dtype=int)

    # Fill the padded array based on the indices in the uneven array
    for i, row in enumerate(uneven_array):
        if i < target_shape[0]:  # Ensure we don't exceed the target number of rows
            for index in row:
                if 0 <= index < target_shape[1]:  # Ensure the index is within bounds
                    padded_array[i, index-1] = 1

    return padded_array

def scale_array(large_array, target_shape):
    # Calculate the scaling factor
    scale_factor = target_shape / large_array.shape[0]

    # Use scipy's zoom function to resize the array
    scaled_array = ndimage.zoom(large_array, (scale_factor, 1), order=1)

    return scaled_array

def hz_to_piano_key(frequency):
    if frequency <= 0:
        return None

    key_number = 12 * math.log2(frequency / 440) + 49
    key_number = round(key_number)

    if 0 <= key_number <= 88:
        return key_number
    else:
        return None


def calculate_metrics(predicted_array, reference_array):
    tp = np.sum((predicted_array == 1) & (reference_array != 0))
    fp = np.sum((predicted_array == 1) & (reference_array == 0))
    fn = np.sum((predicted_array == 0) & (reference_array != 0))

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

def calculate_intersection(correct_array, predicted_array):
    result = np.zeros_like(correct_array)
    result[(correct_array > 0) & (predicted_array > 0)] = 1
    result[(correct_array == 0) & (predicted_array > 0)] = 2

    return result

def plot_midi_with_pitches(midi_array, pitches):
    global pitch_method

    pitches = np.array(pitches)[:]
    midi_array_scaled = scale_array(midi_array, len(pitches))

    if pitch_method == 'siplca':
        pitches_padded = np.zeros_like(pitches)
        pitches_padded[pitches > 0] = 1

        if pitches_padded.shape != midi_array_scaled.shape:
            pitches_padded = np.pad(pitches_padded, ((0, 0), (0, 88 - pitches_padded.shape[1])))
    else:
        pitches_padded = pad_uneven_array(pitches, midi_array_scaled.shape)

    metrics = calculate_metrics(pitches_padded, midi_array_scaled)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(range(midi_array_scaled.shape[0]), np.multiply(np.where(midi_array_scaled > 0, 1, 0), range(1, 89)), marker='.',
             markersize=1, linestyle='', color='blue')
    axs[0].set_ylim(1, 90)
    axs[0].set_title('MIDI')
    axs[0].set_ylabel('Numery dźwięków')

    axs[1].plot(range(pitches_padded.shape[0]), np.multiply(np.where(pitches_padded > 0, 1, 0), range(1, 89)), marker='.',
            markersize=3, linestyle='', color='blue')
    axs[1].set_ylim(1, 90)
    axs[1].set_title('Predykcja')
    axs[1].set_ylabel('Numery dźwięków')

    intersection_array = calculate_intersection(midi_array_scaled, pitches_padded).T

    for value in [0, 1, 2]:
        y, x = np.where(intersection_array == value)
        plt.scatter(x, y, c=[value] * len(x), cmap=ListedColormap(['white', 'green', 'red']),
                    s=3, vmin=0, vmax=2, label=str(value))

        # Customize the plot

    axs[2].set_ylim(1, 90)
    axs[2].set_title(f'Trafność ({CURR_DETECTOR_READABLE_NAME}, F-score = {metrics["F-score"]:.2f})')
    axs[2].set_ylabel('Numery dźwięków')

    fig.tight_layout(h_pad=2.0)
    plt.xlabel('Ramki czasowe')
    # plt.savefig(os.path.join(DATA_CONFIG['save_to'], f'{CURR_TRACK_NAME}__{CURR_DETECTOR_NAME}.png'))
    plt.show()

    print(metrics)

def detect_pitch(beats_detector_function, filename, sr=44100, frame_length=2048, hop_length=512, **kwargs):
    pitches = beats_detector_function(filename, sr, hop_length, frame_length, **kwargs)

    return pitches


if __name__ == '__main__':
    MIDI_RESOLUTION = DATA_CONFIG['midi_resolution']
    TRACK = DATA_CONFIG['tracks'][8]

    CURR_TRACK_NAME = TRACK['name']

    for pitch_method in PITCH_DETECTOR_TO_FILENAME_MAPPERS.keys():
        CURR_DETECTOR_NAME = PITCH_DETECTOR_TO_FILENAME_MAPPERS[pitch_method]
        CURR_DETECTOR_READABLE_NAME = PITCH_DETECTOR_TO_HUMAN_READABLE[pitch_method]

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
        pitch_detector = globals()[f'detect_pitch_{pitch_method}']
        scaling_correction = TRACK['scaling_correction']
        # -------------------------- #

        #    Calculate and plot     #
        pitches = detect_pitch(pitch_detector,
                               audio_track,
                               sample_rate,
                               frame_length=frame_length,
                               hop_length=hop_length,
                               save_pickle=False)

        if pitch_method == 'kt':
            for f, frame in enumerate(pitches):
                pitches[f] = [hz_to_piano_key(hz) for hz in frame]
        elif pitch_method == 'pi2':
            result = []
            for freqs in pitches.values():
                if isinstance(freqs, dict) and 'combo' in freqs:
                    res = []
                    for freq in freqs['combo']:
                        res.append(hz_to_piano_key(freq[0]))

                    result.append(res)
                else:
                    result.append([])
            pitches = result
        elif pitch_method == 'siplca':
            pitches = pitches.T

        midi_array, _ = midi2array(midi_track)

        # scale = midi resolution / time between quarter notes in seconds / frames per second
        scale = MIDI_RESOLUTION / (60 / TRACK_SPEED)  # DP

        # tolerance (number of frames) = 50ms * midi resolution / time between quarter notes in seconds
        tolerance = (TRACK['tolerance_s'] / 1000.) * (MIDI_RESOLUTION / (60 / TRACK_SPEED))

        plot_midi_with_pitches(midi_array, pitches)
        # ------------------------- #
