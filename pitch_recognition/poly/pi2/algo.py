import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from scipy.signal.windows import triang, gaussian
from itertools import combinations
from music21 import pitch

def preprocess_signal(signal, sr, window_size, hop_size, zero_padding_factor=4):
    # Convert ms to samples
    # window_size = int(sr * window_size_ms / 1000)
    # hop_size = int(sr * hop_size_ms / 1000)

    # Zero padding
    n_fft = window_size * zero_padding_factor  # Multiplying the window size by zero padding factor z

    # Compute the magnitude spectrogram using librosa with zero padding
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_size, window='hann', win_length=window_size, center=True)
    Sxx = np.abs(S)  # Get magnitude
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    t = librosa.frames_to_time(np.arange(Sxx.shape[1]), sr=sr, hop_length=hop_size)

    # plot the spectrogram
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.amplitude_to_db(Sxx, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectrogram')
    # plt.tight_layout()
    # plt.show()

    return f, t, Sxx


def find_harmonics(spectrum, frequencies, f0, margin):
    harmonics = []
    f_h = 2*f0
    while f_h < frequencies[-1]:
        lower_bound = f_h - margin
        upper_bound = f_h + margin
        # Find the peak within the margin around the harmonic frequency
        possible_peaks = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
        if possible_peaks.size > 0:
            # Apply a triangular window and find the peak with the maximum weighted value
            window = triang(2 * len(possible_peaks) + 1)[
                     len(possible_peaks) // 2:len(possible_peaks) // 2 + len(possible_peaks)]
            peak_index = possible_peaks[np.argmax(spectrum[possible_peaks] * window)]
            harmonics.append(peak_index)
            f_h = frequencies[peak_index] + f0  # Update the next harmonic frequency base
        else:
            f_h += f0  # No peak found, skip to the next expected harmonic position
    return harmonics


def find_first_harmonics(spectrum, frequencies, candidate, candidate_idx, margin, max_harmonics):
    """Find the first H harmonics for a candidate, considering potential missing ones."""
    f0 = candidate
    f_h = 2 * f0
    harmonics = np.zeros(max_harmonics)  # Store amplitudes, initialized to zeros
    harmonics[0] = candidate_idx # TODO: potencjalnie to usunąć, nie liczyć f0 jako pierwszą harmoniczną

    for i in range(1, max_harmonics):  # first harmonic is the candidate itself TODO: potencjalnie zmienić spowrotem na harmoniczne od 0
        lower_bound = f_h - margin
        upper_bound = f_h + margin
        # Find the peak within the margin around the harmonic frequency
        possible_peaks = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
        if possible_peaks.size > 0:
            # Apply a triangular window and find the peak with the maximum weighted value
            window = triang(2 * len(possible_peaks) + 1)[
                     len(possible_peaks) // 2:len(possible_peaks) // 2 + len(possible_peaks)]
            peak_index = possible_peaks[np.argmax(spectrum[possible_peaks] * window)]
            harmonics[i] = peak_index
            f_h = frequencies[peak_index] + f0  # Update the next harmonic frequency base
        else:
            harmonics[i] = 0
            f_h += f0  # No peak found, skip to the next expected harmonic position

    return harmonics.astype(int)


def candidate_selection(spectrum, frequencies, f_min, f_max, epsilon, margin, max_candidates):
    peaks, properties = find_peaks(spectrum, height=epsilon)
    valid_peaks = peaks[(frequencies[peaks] >= f_min) & (frequencies[peaks] <= f_max)]

    candidates = []
    for peak in valid_peaks:
        f0 = frequencies[peak]
        harmonics = find_harmonics(spectrum, frequencies, f0, margin)
        if harmonics:
            # Sum the amplitudes of harmonics to score the candidate
            amplitude_sum = sum(spectrum[harmonics])
            candidates.append((f0, amplitude_sum, peak))

    # Sort candidates based on the sum of their harmonics' amplitudes
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_candidates = candidates[:max_candidates]  # Select top F candidates

    return selected_candidates


def generate_combinations(candidates, max_polyphony):
    # Generate all combinations of candidates for polyphonies from 1 to max_polyphony
    all_combinations = []
    for r in range(1, max_polyphony + 1):
        all_combinations.extend(combinations(candidates, r))
    return all_combinations


def evaluate_combo(combo, spectra, frequencies, margin, max_harmonics):
    combo_score = 0
    combo_pattern = {}
    for candidate in combo:
        harmonic_indices = find_harmonics(spectra, frequencies, candidate, margin)
        harmonic_amplitudes = spectra[harmonic_indices[:max_harmonics]]
        # Assuming linear additivity and simplicity for combining harmonic amplitudes
        combo_score += np.sum(harmonic_amplitudes)
        combo_pattern[candidate] = harmonic_amplitudes

    return combo_score, combo_pattern


def handle_overlapping_partials(all_candidates_harmonics_names, combo_spectra, harmonic_amplitudes, harmonic_names, cand_idx):
    """Adjust harmonics in case of overlaps using interpolation and modify spectra residuals."""
    start_from = None

    for i, harmonic_name in enumerate(harmonic_names):
        if i == 0 or i == len(harmonic_names) - 1:
            continue

        for j, candidate_harmonic_names in enumerate(all_candidates_harmonics_names[cand_idx + 1:]):
            if harmonic_name in candidate_harmonic_names:
                if start_from is None:
                    start_from = i
                break

        else:
            if start_from is not None:
                interp_values = np.interp(np.arange(start_from, i), [start_from-1, i], [harmonic_amplitudes[start_from-1], harmonic_amplitudes[i]])
                curr_amplitudes = harmonic_amplitudes[start_from:i]

                curr_names = harmonic_names[start_from:i]

                for k, curr_amp in enumerate(curr_amplitudes):
                    for j, candidate_harmonic_names in enumerate(all_candidates_harmonics_names[cand_idx + 1:]):
                        if curr_names[k] in candidate_harmonic_names:
                            adj_cand_name_idx = candidate_harmonic_names.index(curr_names[k])

                            if curr_amp < interp_values[k]:
                                combo_spectra[j + cand_idx + 1][adj_cand_name_idx] = 0
                            else:
                                combo_spectra[j + cand_idx + 1][adj_cand_name_idx] -= interp_values[k]
                                combo_spectra[j + cand_idx + 1][adj_cand_name_idx] = max(0, combo_spectra[j + cand_idx + 1][adj_cand_name_idx])

                harmonic_amplitudes[start_from:i] = [min(pair) for pair in zip(interp_values, curr_amplitudes)]
                start_from = None

    if start_from is not None:
        interp_values = np.interp(np.arange(start_from, i), [start_from - 1, i],
                                  [harmonic_amplitudes[start_from - 1], harmonic_amplitudes[i]])
        curr_amplitudes = harmonic_amplitudes[start_from:i]

        curr_names = harmonic_names[start_from:i]

        for k, curr_amp in enumerate(curr_amplitudes):
            for j, candidate_harmonic_names in enumerate(all_candidates_harmonics_names[cand_idx + 1:]):
                if curr_names[k] in candidate_harmonic_names:
                    adj_cand_name_idx = candidate_harmonic_names.index(curr_names[k])

                    if curr_amp < interp_values[k]:
                        combo_spectra[j + cand_idx + 1][adj_cand_name_idx] = 0
                    else:
                        combo_spectra[j + cand_idx + 1][adj_cand_name_idx] -= interp_values[k]
                        combo_spectra[j + cand_idx + 1][adj_cand_name_idx] = max(0, combo_spectra[j + cand_idx + 1][
                            adj_cand_name_idx])

        harmonic_amplitudes[start_from:i] = [min(pair) for pair in zip(interp_values, curr_amplitudes)]

    return harmonic_amplitudes


def evaluate_candidate_smoothness(harmonics, max_index):
    """Evaluate the smoothness of a candidate's harmonic pattern."""
    if max_index == 0:
        return 0  # No harmonics to evaluate smoothness
    p_bar = harmonics / np.max(harmonics)  # Normalize
    gaussian_window = [0.21, 0.58, 0.21]  # Truncated Gaussian window
    p_tilde = convolve(p_bar, gaussian_window, mode='same') / np.sum(gaussian_window)

    roughness = np.sum(np.abs(p_tilde - p_bar))
    normalized_roughness = roughness / (1 - np.max(gaussian_window))
    smoothness = 1 - (normalized_roughness / max_index)

    return smoothness

def evaluate_combinations(combinations, spectra, frequencies, margin, max_harmonics, kappa, frame_idx):
    global frame_scores
    frame_scores[frame_idx] = []

    for i in range(len(combinations)):
        combinations[i] = sorted(combinations[i], key=lambda x: x[2])

    # best_score = 0
    # best_combination = None

    for j, combo in enumerate(combinations):
        combo_scores = []

        combo_harmonic_indices = [find_first_harmonics(spectra, frequencies, candidate, idx, margin, max_harmonics) for candidate, amp, idx in combo]
        combo_harmonic_names = [frequency_to_pitch([frequencies[harmonic_idx] for harmonic_idx in harmonic_indices], False) for harmonic_indices in combo_harmonic_indices]
        combo_spectra = [spectra[harmonic_indices] for harmonic_indices in combo_harmonic_indices]

        for i, (candidate, _, _) in enumerate(combo):
            current_candidate_harmonic_indices = combo_harmonic_indices[i]
            harmonic_names = combo_harmonic_names[i]
            harmonic_amplitudes = combo_spectra[i]
            harmonic_amplitudes = handle_overlapping_partials(combo_harmonic_names, combo_spectra, harmonic_amplitudes, harmonic_names, i)

            intensity = np.sum(harmonic_amplitudes)
            smoothness = evaluate_candidate_smoothness(harmonic_amplitudes, len(current_candidate_harmonic_indices))
            combo_scores.append(score_candidate(intensity, smoothness, kappa))

        combo_notes = frequency_to_pitch([candidate for candidate, _, _ in combo])

        # TODO: odejmowanie części wyniku o 7% za każdą dodatkową nutę w akordzie całkiem nieźle poprawia algorytm
        for k, frame_data in enumerate(frame_scores[frame_idx]):
            if frame_data['combo_notes'] == combo_notes:
                if np.sum(combo_scores) * (1 - (0.07 * (len(frame_data['combo_notes']) - 1))) > frame_data['score']: # TODO: zmienić spowrotem na np.sum(combo_scores)
                    frame_scores[frame_idx][k]['score'] = np.sum(combo_scores) * (1 - (0.07 * (len(frame_data['combo_notes']) - 1))) # TODO: zmienić spowrotem na np.sum(combo_scores)
                break
        else:
            frame_scores[frame_idx].append({'combo': tuple(combo), 'combo_notes': combo_notes, 'score': np.sum(combo_scores) * (1 - (0.07 * (len(combo_notes) - 1)))}) # TODO: zmienić spowrotem na np.sum(combo_scores)

    #     combo_total_score = np.sum([score ** 2 for score in combo_scores])  # Square to emphasize higher scores
    #     if combo_total_score > best_score:
    #         best_score = combo_total_score
    #         best_combination = combo
    #
    # return best_combination, best_score


def temporal_smoothing(frame_scores, num_frames, K):
    # Apply temporal smoothing
    smoothed_scores = {}
    for frame_index in range(num_frames):
        smoothed_scores[frame_index] = []
        for offset in range(-K, K + 1):
            adjacent_index = frame_index + offset
            if 0 <= adjacent_index < num_frames:
                for adjacent_frame_data in frame_scores[adjacent_index]:
                    for it, smoothed_score_data in enumerate(smoothed_scores[frame_index]):
                        if adjacent_frame_data['combo_notes'] == smoothed_score_data['combo_notes']:
                            smoothed_scores[frame_index][it]['score'] += adjacent_frame_data['score']
                            break
                    else:
                        smoothed_scores[frame_index].append({
                            'combo': adjacent_frame_data['combo'],
                            'combo_notes': adjacent_frame_data['combo_notes'],
                            'score': adjacent_frame_data['score']
                        })

    # Determine best combination for each frame
    best_combinations = {}
    for frame_index in range(num_frames):
        best_combinations[frame_index] = max(smoothed_scores[frame_index], key=lambda item: item['score'], default=(None, 0))

    return best_combinations


def frequency_to_pitch(frequencies, sort = True):
    """Convert a list of frequencies to musical pitches using librosa."""
    if sort:
        return tuple(sorted(librosa.hz_to_note(f) for f in frequencies))

    return tuple(librosa.hz_to_note(f) for f in frequencies)


def score_candidate(intensity, smoothness, kappa=1):
    """Compute the candidate score based on intensity and smoothness."""
    return intensity * (smoothness ** kappa)


def detect_pitch_pi2(filename, fs=44100, window_size=2048, hop_size=256, **kwargs):
    audio, fs = librosa.load(filename, sr=fs)
    f_min, f_max = 50, 2000
    epsilon = 0.5
    margin = 20  # Frequency margin for inharmonicity
    max_candidates = 5  # Max number of candidates to select
    max_harmonics = 7
    max_polyphony = 1
    kappa = 1.5  # Weight for the smoothness evaluation
    K = 1  # take K frames around the current frame into account
    global frame_scores
    frame_scores = {}

    f, t, Sxx = preprocess_signal(audio, fs, window_size, hop_size)

    for i in range(Sxx.shape[1]):
        print(f"Processing frame {i} of {Sxx.shape[1]}...")
        selected_candidates = candidate_selection(Sxx[:, i], f, f_min, f_max, epsilon, margin, max_candidates)
        combinations_arr = generate_combinations(selected_candidates, max_polyphony)
        evaluate_combinations(combinations_arr, Sxx[:, i], f, margin, max_harmonics, kappa, i)

    best_combinations = temporal_smoothing(frame_scores, Sxx.shape[1], K)

    return best_combinations


# Example usage
if __name__ == "__main__":
    best_combinations = detect_pitch_pi2('../../../audio/midi_tracks/Triada_C.mp3', fs=44100, window_size=2048, hop_size=256)

    print(best_combinations[0])

