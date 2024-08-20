import os
import pickle

import numpy as np
import soundfile as sf
from scipy import ndimage
from scipy.signal import argrelextrema
from sporco.admm import cbpdn
from sporco import util
import librosa
import matplotlib.pyplot as plt
from definitions import ROOT_DIR


def learn_dictionary(audio_folder, num_notes=88, note_length=1, sample_rate=11025):
    dictionary = []

    for i in range(num_notes):
        note_file = os.path.join(audio_folder, f'note_{i + 21}.wav')  # Assuming MIDI note numbers start at 21 (A0)
        if os.path.exists(note_file):
            audio, sr = librosa.load(note_file, sr=sample_rate, duration=note_length, mono=True)

            # Ensure the audio is the right length
            if len(audio) < note_length * sample_rate:
                audio = np.pad(audio, (0, note_length * sample_rate - len(audio)))
            else:
                audio = audio[:note_length * sample_rate]

            dictionary.append(audio)
        else:
            print(f"Warning: File not found for note {i + 21}")
            dictionary.append(np.zeros(note_length * sample_rate))

    return np.array(dictionary)
    pass


def transcribe_audio(audio_file, dictionary, sample_rate=11025, lambda_val=0.05, **kwargs):
    global instrument

    filename = os.path.basename(audio_file).split('.')[0]
    pickle_filename = os.path.join(ROOT_DIR, 'pitch_recognition/poly/cbpdn', f'{instrument}_results_{filename}.pkl')

    if os.path.exists(pickle_filename):
        print(f'{instrument} results found for {filename}. Loading {instrument} results and passing problem solving...')
        with open(pickle_filename, 'rb') as fid:
            X = pickle.load(fid)
        return X

        # Load the audio file
    audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

    # ConvBPDN expects the dictionary to have notes as columns
    dictionary = dictionary.T

    # Set up the CBPDN problem
    print('Setting up the CBPDN problem...')
    opt = cbpdn.ConvBPDN.Options({'Verbose': True,
                                  'MaxMainIter': 1000,
                                  'RelStopTol': 1e-5,
                                  'HighMemSolve': True,
                                  'LinSolveCheck': False,
                                  'AuxVarObj': False,
                                  'AutoRho': {'Enabled': True}})

    # Solve the CBPDN problem
    print('Solving the CBPDN problem...')
    b = cbpdn.ConvBPDN(dictionary, audio, lambda_val, opt, dimN=1)
    X = b.solve()

    # flatten extra dimensions of array for easier peak-picking
    X = X[:, 0, 0, :]

    # Save the results
    print(f'Saving {instrument} results...')
    if 'save_pickle' in kwargs:
        with open(os.path.join(ROOT_DIR, 'pitch_recognition/poly/cbpdn', f'{instrument}_results_{filename}.pkl'),
                  'wb') as fid:
            pickle.dump(X, fid)

    return X


def post_process_and_peak_pick(Y, sr, window_size=512):
    # Find local maxima at each timestep, ignoring negative peaks
    lmaxes = np.concatenate([row[argrelextrema(row, np.greater)[0]] for row in Y])

    # Calculate q3 and q1
    p75, p25 = np.percentile(lmaxes, [75, 25])

    # Pick the binarization threshold based on the inter-quartile range
    # todo: do all the melodies with noisy thresholding and clear thresholding, e.g. for lot trzemiala its 0.005 for clear and 0.001 for noisy
    threshold = p75 + 8 * (p75 - p25)

    # Binarize
    Y[Y < threshold] = 0
    print(f"Threshold: {threshold}")

    # Initialize variables for parsing 50ms windows
    l = int(window_size)
    num_windows = int(Y.shape[0] / window_size)

    # 2D array that represents onsets
    onsets = []

    for window in range(num_windows):
        start = window * l
        end = (window + 1) * l
        row = Y[start:end, :]
        sumrow = np.sum(row, axis=0)
        relex = argrelextrema(sumrow, np.greater)[0]
        onsets.append(relex + 1)

    return onsets


def onsets_to_notes(onsets, sr, window_size=512):
    notes = []
    for note in range(onsets.shape[0]):
        note_onsets = np.where(onsets[note, :] == 1)[0]
        for onset in note_onsets:
            notes.append((note, onset * (window_size / sr)))
    return sorted(notes, key=lambda x: x[1])


def detect_pitch_cbpdn(filename, sr=11025, hop_length=128, window_size=256, **kwargs):
    global instrument
    instrument = 'piano'

    # Load dictionary
    dictionary = np.load(
        os.path.join(
            ROOT_DIR,
            f'audio/{instrument}/remastered_v2/extracted_notes_1sec/{instrument}_all_notes_1sec_normalized.npy'
        )
    )

    # Transcribe audio
    X = transcribe_audio(filename, dictionary, sr, 0.05, **kwargs)

    # Post-process and peak pick the results
    onsets = post_process_and_peak_pick(X, sr=sr, window_size=window_size)

    # Convert onsets to MIDI notes and times notes = onsets_to_notes(onsets, sr, window_size) Plot the results    #
    # plt.figure(figsize=(12, 6))
    # for note, time in notes:
    #     plt.plot(time, note, 'ro', markersize=5)
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('MIDI Note Number')
    # plt.title('Transcription Results')
    #
    # plt.show()
    return onsets


if __name__ == "__main__":
    global instrument

    audio_file = os.path.join(ROOT_DIR, f'audio/wlazl_kotek/wlazl_kotek.mp3')

    detect_pitch_cbpdn(audio_file, sr=11025, hop_length=128, window_size=512, save_pickle=True)