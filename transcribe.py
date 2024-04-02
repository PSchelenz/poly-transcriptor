import matplotlib.pyplot as plt
import numpy as np
import os

from pitch_recognition.poly.si_plca.si_plca_algo import transcription

from oto_recognition.poly.spectral_flux import detect_onsets as spectral_flux_onset_detection
from oto_recognition.poly.hfc import onset_detection_hfc
from oto_recognition.poly.rnn import onset_detection_rnn
from oto_recognition.poly.phase_deviation import onset_detection_phase_deviation
from oto_recognition.poly.spectral_diff import onset_detection_spectral_diff
from oto_recognition.poly.complex_domain import onset_detection_complex_domain

from rhythm_recognition.poly.dbn import detect_beat_dbn
from rhythm_recognition.poly.crf import detect_beat_crf
from rhythm_recognition.poly.dynamic_programming import dynamic_programming_beat_recognition

def refine_notes(piano_roll, onsets, beats):
    print('Refining piano roll...')
    refined_piano_roll = np.zeros(piano_roll.shape)  # moved notes to the nearest onsets
    hyper_refined_piano_roll = np.zeros(piano_roll.shape)  # no

    onsets_roll = np.zeros(piano_roll.shape[1])
    onsets_roll[np.round(onsets).astype(int)] = 1

    beats_roll = np.zeros(piano_roll.shape[1])
    beats_roll[np.round(beats).astype(int)] = 1

    plt.pcolormesh(np.arange(piano_roll.shape[1]), np.arange(piano_roll.shape[0]),
                   piano_roll,
                   shading='auto', cmap='YlOrBr', alpha=0.5)

    skip_frames = 0
    for r, row in enumerate(piano_roll):  # move pitches to the nearest onsets
        for i, frame_val in enumerate(row):
            if skip_frames:
                skip_frames -= 1
                continue

            skip_frames = 0
            if frame_val > 0:  # note start detected
                skip_frames = 1
                for next_frame_idx in range(i + 1, len(row)):  # count note duration in frames
                    if row[next_frame_idx] == 0:
                        break
                    skip_frames += 1

                if onsets_roll[i] != 1:  # if at the beginning of the note there is no onset, find the nearest onset
                    nearest_onset_idx = find_nearest_onset(onsets_roll, i)

                    if i < nearest_onset_idx < i + skip_frames:  # if closest onset is in the middle of a note, cut the beginning of the note until the onset
                        refined_piano_roll[r, nearest_onset_idx : next_frame_idx] = 1
                    else:
                        refined_piano_roll[r, nearest_onset_idx:nearest_onset_idx + skip_frames] = 1  # otherwise just move the note to the left
                else:
                    refined_piano_roll[r, i:i + skip_frames] = 1

                skip_frames -= 1  # correction for the loop

    shortest_note_duration = 2  # 1 - quarter note, 2 - eighth note, 4 - sixteenth note etc.
    shortest_note_indexes = find_valid_indexes_for_note_duration(beats_roll, shortest_note_duration)

    for r, row in enumerate(refined_piano_roll):
        for i, frame_val in enumerate(row):
            if skip_frames:
                skip_frames -= 1
                continue

            skip_frames = 0
            if frame_val > 0:
                for next_frame_idx in range(i + 1, len(row)):
                    if row[next_frame_idx] == 0:
                        break
                    skip_frames += 1

                if beats_roll[i] != 1:
                    closest_start_beat_frame = find_nearest_beat(shortest_note_indexes, i)  # find nearest beat based on chosen shortest note duration

                    if beats_roll[closest_start_beat_frame + skip_frames - 1] != 1:
                        closest_end_beat_frame = find_nearest_beat_to_the_left(shortest_note_indexes, closest_start_beat_frame + skip_frames - 1)

                        # don't allow notes to start and end at the same beat
                        if closest_start_beat_frame == closest_end_beat_frame:
                            index_of_end_frame_in_available_indexes = np.where(shortest_note_indexes == closest_end_beat_frame)
                            new_index = index_of_end_frame_in_available_indexes[0][0] + 1

                            if new_index >= len(shortest_note_indexes):
                                closest_end_beat_frame = len(row) - 1
                            else:
                                closest_end_beat_frame = shortest_note_indexes[new_index]

                        hyper_refined_piano_roll[r, closest_start_beat_frame:closest_end_beat_frame + 1] = 1
                    else:
                        hyper_refined_piano_roll[r, closest_start_beat_frame:closest_start_beat_frame + skip_frames] = 1
                else:
                    if beats_roll[i + skip_frames] != 1:
                        closest_end_beat_frame = find_nearest_beat_to_the_left(shortest_note_indexes, i + skip_frames - 1)

                        # don't allow notes to start and end at the same beat
                        if i == closest_end_beat_frame:
                            index_of_end_frame_in_available_indexes = np.where(shortest_note_indexes == closest_end_beat_frame)
                            new_index = index_of_end_frame_in_available_indexes[0][0] + 1

                            if new_index >= len(shortest_note_indexes):
                                closest_end_beat_frame = len(row) - 1
                            else:
                                closest_end_beat_frame = shortest_note_indexes[new_index]

                        hyper_refined_piano_roll[r, i:closest_end_beat_frame + 1] = 1

    plt.pcolormesh(np.arange(hyper_refined_piano_roll.shape[1]), np.arange(hyper_refined_piano_roll.shape[0]),
                   hyper_refined_piano_roll,
                   shading='auto', cmap='binary')

def find_nearest_onset(onsets_roll, i):
    left_steps = np.inf
    right_steps = np.inf

    for j in range(i, 0, -1):  # travel to the left
        if onsets_roll[j] == 1:
            left_steps = i - j
            break

    for j in range(i, len(onsets_roll)):  # travel to the right
        if onsets_roll[j] == 1:
            right_steps = j - i
            break

    if left_steps < right_steps:
        return i - left_steps

    return i + right_steps

def find_nearest_beat(valid_indexes, i):
    closest_index = min(valid_indexes, key=lambda x: abs(x - i))

    return closest_index

def find_nearest_beat_to_the_left(valid_indexes, i):
    return max(filter(lambda x: x < i, valid_indexes))

def find_valid_indexes_for_note_duration(beats_roll, note_duration):
    valid_indexes = []
    last_index = None
    divisions = []

    for i, _ in enumerate(beats_roll):
        if beats_roll[i] > 0:
            valid_indexes.append(np.int64(i))
            current_index = i

            if last_index is not None:
                note_duration_dependent_indexes = np.linspace(last_index, current_index, note_duration, endpoint=False)
                valid_indexes.extend(np.round(note_duration_dependent_indexes[1:]).astype(int))

                divisions.append((current_index - last_index) / note_duration)

            last_index = current_index

    mean_division = np.mean(divisions)

    for i in range(last_index + 1, len(beats_roll)):
        if i % round(mean_division) == 0:
            valid_indexes.append(np.int64(i))

    return sorted(valid_indexes)

audio_file = os.path.abspath('audio/midi_tracks/Canon_in_D.mp3')
sr = 44100
plt.figure(figsize=(14, 4))

piano_roll = transcription(audio_file, 50, 3, 1.18, 1.15, 1, 'pitch_recognition/poly/si_plca/shiftedW.mat', sr, 1, 60, 0.022, 6)
onsets = onset_detection_complex_domain(audio_file, sr, 60)
beats = detect_beat_crf(audio_file, 248, 60)

refine_notes(piano_roll, onsets, beats)

plt.show()