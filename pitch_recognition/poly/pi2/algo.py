import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from scipy.signal.windows import triang, gaussian
from itertools import combinations


def preprocess_signal(signal, sr, window_size_ms=93, hop_size_ms=9.28, zero_padding_factor=4):
    # Convert ms to samples
    window_size = int(sr * window_size_ms / 1000)
    hop_size = int(sr * hop_size_ms / 1000)

    # Zero padding
    n_fft = window_size * zero_padding_factor  # Multiplying the window size by zero padding factor z

    # Compute the magnitude spectrogram using librosa with zero padding
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_size, window='hann', win_length=window_size, center=True)
    Sxx = np.abs(S)  # Get magnitude
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    t = librosa.frames_to_time(np.arange(Sxx.shape[1]), sr=sr, hop_length=hop_size)

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


def find_first_harmonics(spectrum, frequencies, candidate, margin, max_harmonics):
    """Find the first H harmonics for a candidate, considering potential missing ones."""
    f0 = frequencies[candidate]
    harmonics = np.zeros(max_harmonics)  # Store amplitudes, initialized to zero

    for i in range(max_harmonics):
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
            harmonics.append(0)
            f_h += f0  # No peak found, skip to the next expected harmonic position

    return harmonics


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
            candidates.append((f0, amplitude_sum))

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


def handle_overlapping_partials(all_candidates_harmonics, harmonic_amplitudes, harmonic_indices, spectra, cand_idx):
    """Adjust harmonics in case of overlaps using interpolation and modify spectra residuals."""
    start_from = None

    for i, harmonic_index in enumerate(harmonic_indices):
        if i == 0 or i == len(harmonic_indices) - 1:
            continue

        for j, candidates_harmonics in enumerate(all_candidates_harmonics):
            if j != cand_idx and harmonic_index in candidates_harmonics:
                start_from = i - 1
                break
        else:
            if start_from is not None:
                interp_values = np.interp(np.arange(start_from, i + 1), [start_from, i], [harmonic_amplitudes[start_from], harmonic_amplitudes[i]])
                harmonic_amplitudes[start_from:i + 1] = interp_values
                start_from = None

    return harmonic_amplitudes # TODO: moja zmiana, sprawdzić


def evaluate_candidate_smoothness(harmonics, max_index):
    """Evaluate the smoothness of a candidate's harmonic pattern."""
    if max_index == 0:
        return 0  # No harmonics to evaluate smoothness
    p_bar = harmonics / np.max(harmonics)  # Normalize
    gaussian_window = gaussian(3, std=1)  # Truncated Gaussian window
    p_tilde = convolve(p_bar, gaussian_window, mode='same') / np.sum(gaussian_window)

    roughness = np.sum(np.abs(p_tilde - p_bar))
    normalized_roughness = roughness / (1 - np.max(gaussian_window))
    smoothness = 1 - (normalized_roughness / max_index)

    return smoothness

def evaluate_combinations(combinations, spectra, frequencies, margin, max_harmonics, kappa):
    best_score = 0
    best_combination = None

    for combo in combinations:
        combo_scores = []

        combo_harmonic_indices = [find_first_harmonics(spectra, frequencies, candidate, margin, max_harmonics) for candidate, _ in combo]

        for i, (candidate, _) in enumerate(combo):
            current_candidate_harmonic_indices = combo_harmonic_indices[i]
            harmonic_amplitudes = spectra[current_candidate_harmonic_indices]
            harmonic_amplitudes = handle_overlapping_partials(combo_harmonic_indices, harmonic_amplitudes, current_candidate_harmonic_indices, spectra, i)

            intensity = np.sum(harmonic_amplitudes)
            smoothness = evaluate_candidate_smoothness(harmonic_amplitudes, len(current_candidate_harmonic_indices))
            combo_scores.append(score_candidate(intensity, smoothness, kappa))

        combo_total_score = np.sum([score ** 2 for score in combo_scores])  # Square to emphasize higher scores
        if combo_total_score > best_score:
            best_score = combo_total_score
            best_combination = combo

    return best_combination, best_score


def score_candidate(intensity, smoothness, kappa=1):
    """Compute the candidate score based on intensity and smoothness."""
    return intensity * (smoothness ** kappa)


# Example usage
if __name__ == "__main__":
    audio, fs = librosa.load('../../../audio/midi_tracks/Canon_in_D.mp3', sr=None, duration=2, offset=0.2)
    f_min, f_max = 50, 2000
    epsilon = 0.1
    margin = 10  # Frequency margin for inharmonicity
    max_candidates = 5  # Max number of candidates to select
    max_harmonics = 5
    max_polyphony = 3
    kappa = 1  # Weight for the smoothness evaluation

    f, t, Sxx = preprocess_signal(audio, fs)
    selected_candidates = candidate_selection(Sxx[:, 0], f, f_min, f_max, epsilon, margin, max_candidates)
    combinations = generate_combinations(selected_candidates, max_polyphony)
    best_combination, best_score = evaluate_combinations(combinations, Sxx[:, 0], f, margin, max_harmonics, kappa)

    print(f"Best combination: {best_combination}")
    print(f"Best score: {best_score}")