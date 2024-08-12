import numpy as np
from scipy.signal import stft, find_peaks
from scipy.fftpack import fft
import itertools
import librosa
import math


def preprocess(audio, sr, frame_size=4096, hop_length=512, z_pad_factor=4):
    # Compute STFT with zero-padding
    n_fft = frame_size * z_pad_factor
    f, t, S = stft(audio, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_length, nfft=n_fft)
    return f, t, np.abs(S)


def select_peaks(spectrum, f, height=0.1, distance=None):
    # Find peaks using scipy.signal.find_peaks
    peak_indices, _ = find_peaks(spectrum, height=height, distance=distance)

    # Create list of (frequency, amplitude) tuples for the detected peaks
    peaks = [(f[i], spectrum[i]) for i in peak_indices]

    return peaks


def select_f0_candidates(peaks, f_min=38, f_max=2100, max_candidates=10, epsilon=2):
    candidates = [p for p in peaks if f_min <= p[0] <= f_max and p[1] > epsilon]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_candidates]


def find_harmonics(f0, spectrum, f, num_harmonics=10, fr=11):
    harmonics = []
    for h in range(1, num_harmonics + 1):
        target_f = f0 * h
        idx_range = np.where((f >= target_f - fr) & (f <= target_f + fr))[0]
        if len(idx_range) > 0:
            peak_idx = idx_range[np.argmax(spectrum[idx_range])]
            harmonics.append((f[peak_idx], spectrum[peak_idx]))
        else:
            harmonics.append((target_f, 0))
    return harmonics


def generate_combinations(candidates, max_polyphony=6):
    combinations = []
    for i in range(1, min(len(candidates), max_polyphony) + 1):
        combinations.extend(itertools.combinations(candidates, i))
    return combinations


def infer_hps(combination, spectrum, f, num_harmonics=10):
    hps_list = []
    for f0, _ in combination:
        harmonics = find_harmonics(f0, spectrum, f, num_harmonics)
        hps = np.zeros(num_harmonics)
        for i, (_, amp) in enumerate(harmonics):
            hps[i] = amp
        hps_list.append(hps)

    # Handle overlapping partials
    for i in range(len(combination)):
        for j in range(i + 1, len(combination)):
            for h in range(num_harmonics):
                if abs(combination[i][0] * (h + 1) - combination[j][0] * (h + 1)) < 1:  # Adjust threshold as needed
                    # Estimate amplitudes by linear interpolation
                    if h > 0 and h < num_harmonics - 1:
                        hps_list[i][h] = (hps_list[i][h - 1] + hps_list[i][h + 1]) / 2
                        hps_list[j][h] = (hps_list[j][h - 1] + hps_list[j][h + 1]) / 2

    return hps_list


def spectral_smoothness(hps):
    hps_normalized = hps / np.max(hps)
    gaussian_window = np.array([0.21, 0.58, 0.21])
    smoothed = np.convolve(hps_normalized, gaussian_window, mode='same')
    roughness = np.sum(np.abs(smoothed - hps_normalized))
    smoothness = 1 - (roughness / len(hps))
    return smoothness


def evaluate_candidate(hps, kappa=2):
    intensity = np.sum(hps)
    smoothness = spectral_smoothness(hps)
    return intensity * (smoothness ** kappa)


def evaluate_combination(hps_list, gamma=0.1, eta=5):
    scores = [evaluate_candidate(hps) for hps in hps_list]
    total_intensity = sum(np.sum(hps) for hps in hps_list)
    max_intensity = max(np.sum(hps) for hps in hps_list)

    # Apply intensity thresholds
    if total_intensity < eta or any(np.sum(hps) < gamma * max_intensity for hps in hps_list):
        return 0

    return sum(score ** 2 for score in scores)


def select_best_combination(combinations, spectrum, f):
    best_score = -np.inf
    best_combination = None
    for combination in combinations:
        hps_list = infer_hps(combination, spectrum, f)
        score = evaluate_combination(hps_list)
        if score > best_score:
            best_score = score
            best_combination = combination
    return best_combination


def temporal_smoothing(frame_combinations, K=2):
    smoothed_combinations = []
    for t in range(len(frame_combinations)):
        start = max(0, t - K)
        end = min(len(frame_combinations), t + K + 1)
        neighbor_combinations = frame_combinations[start:end]

        # Count occurrences of each pitch in the neighborhood
        pitch_counts = {}
        for combo in neighbor_combinations:
            for f0, _ in combo:
                pitch = round(f0)  # Round to nearest Hz for simplicity
                pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

        # Select pitches that appear in more than half of the frames
        threshold = len(neighbor_combinations) / 2
        smoothed_combo = [(f0, _) for f0, _ in frame_combinations[t] if pitch_counts[round(f0)] > threshold]
        smoothed_combinations.append(smoothed_combo)

    return smoothed_combinations


def pitch_tracking(smoothed_combinations, t, f):
    # Simplified pitch tracking using a basic continuity constraint
    tracked_pitches = []
    for i in range(1, len(smoothed_combinations)):
        prev_combo = smoothed_combinations[i - 1]
        curr_combo = smoothed_combinations[i]

        for prev_f0, _ in prev_combo:
            closest_f0 = min(curr_combo, key=lambda x: abs(x[0] - prev_f0), default=None)
            if closest_f0 and abs(closest_f0[0] - prev_f0) < 10:  # Adjust threshold as needed
                tracked_pitches.append((t[i], closest_f0[0]))

    return tracked_pitches


def detect_pitch_pi2(filename, fs=44100, hop_size=256, frame_size=2048, **kwargs):
    # Load audio file
    audio, sr = librosa.load(filename, sr=fs)

    # Parameters
    params = {
        "preprocess": {
            "frame_size": frame_size,
            "hop_length": hop_size,
            "z_pad_factor": 4
        },
        "peaks_selection": {
            "height": 0.1,
            "distance": None,
        },
        "candidates_selection": {
            "f_min": 38,
            "f_max": 2100,
            "max_candidates": 10,
            "epsilon": 2,
        },
        "combinations_generation": {
            "max_polyphony": 5,
        },
        "temporal_smoothing": {
            "K": 2,
        }
    }

    # Preprocess
    f, t, S = preprocess(audio, sr, **params['preprocess'])

    frame_combinations = []
    for i in range(S.shape[1]):  # For each time frame
        spectrum = S[:, i]

        # Select peaks
        peaks = select_peaks(spectrum, f, **params['peaks_selection'])

        # Select f0 candidates
        candidates = select_f0_candidates(peaks, **params['candidates_selection'])

        # Generate combinations
        combinations = generate_combinations(candidates, **params['combinations_generation'])

        # Evaluate combinations and select the best
        best_combination = select_best_combination(combinations, spectrum, f)

        frame_combinations.append(best_combination)

    # Temporal smoothing
    smoothed_combinations = temporal_smoothing(frame_combinations, **params['temporal_smoothing'])

    # Pitch tracking
    tracked_pitches = pitch_tracking(smoothed_combinations, t, f)

    return tracked_pitches

def hz_to_note(frequency):
    return librosa.hz_to_note(frequency)


if __name__ == '__main__':
    # Example usage
    audio_file = "../../../audio/wlazl_kotek/wlazl_kotek.mp3"
    f0_estimates = detect_pitch_pi2(audio_file, 44100)

    # Print results
    for time, f0 in f0_estimates:
        print(f"Time: {time:.2f}s, F0: {f0:.2f}Hz, Piano note:{hz_to_note(f0)}")
