import math

import numpy as np
from pitch_recognition.configurator import load_audio


def preprocess_signal(signal, fs):
    # Normalize the signal
    signal = signal / np.max(np.abs(signal))

    # Apply pre-emphasis filter
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    return emphasized_signal


def compute_acf(signal, max_lag):
    # Compute autocorrelation function
    acf = np.correlate(signal, signal, mode='full')
    acf = acf[len(acf) // 2:len(acf) // 2 + max_lag]
    return acf / acf[0]


def estimate_f0(acf, fs, min_f0, max_f0):
    min_lag = int(fs / max_f0)
    max_lag = int(fs / min_f0)

    # Find peaks in ACF
    peaks = np.array([i for i in range(min_lag, max_lag) if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]])

    if len(peaks) == 0:
        return None

    # Find the highest peak
    best_peak = peaks[np.argmax(acf[peaks])]

    return fs / best_peak


# def karjalainen_tolonen_f0(signal, fs, frame_length=2048, hop_length=512, min_f0=50, max_f0=500):
#     # Preprocess the signal
#     preprocessed_signal = preprocess_signal(signal, fs)
#
#     # Frame the signal
#     frames = np.array([preprocessed_signal[i:i + frame_length] for i in
#                        range(0, len(preprocessed_signal) - frame_length, hop_length)])
#
#     f0_estimates = []
#
#     for frame in frames:
#         acf = compute_acf(frame, frame_length // 2)
#         f0 = estimate_f0(acf, fs, min_f0, max_f0)
#         f0_estimates.append(f0)
#
#     return f0_estimates


def generate_harmonic_signal(f0, fs, duration):
    t = np.arange(0, duration, 1 / fs)
    signal = np.zeros_like(t)
    for i in range(1, 6):  # Generate 5 harmonics
        signal += np.sin(2 * np.pi * i * f0 * t) / i
    return signal


def cancel_harmonic(signal, f0, fs):
    duration = len(signal) / fs
    harmonic_signal = generate_harmonic_signal(f0, fs, duration)

    # Normalize and align the harmonic signal
    harmonic_signal = harmonic_signal[:len(signal)]  # Ensure same length
    harmonic_signal = harmonic_signal / np.max(np.abs(harmonic_signal))

    # Find the optimal scaling factor
    scale = np.dot(signal, harmonic_signal) / np.dot(harmonic_signal, harmonic_signal)

    # Subtract the scaled harmonic signal
    cancelled_signal = signal - scale * harmonic_signal

    return cancelled_signal


def estimate_multiple_f0(signal, fs, num_pitches=3, min_f0=50, max_f0=500):
    remaining_signal = signal.copy()
    f0_estimates = []

    for _ in range(num_pitches):
        acf = compute_acf(remaining_signal, len(remaining_signal) // 2)
        f0 = estimate_f0(acf, fs, min_f0, max_f0)

        if f0 is None:
            break

        f0_estimates.append(f0)
        remaining_signal = cancel_harmonic(remaining_signal, f0, fs)

    return f0_estimates


def detect_pitch_kt(filename, fs=44100, hop_length=512, frame_length=2048, num_pitches=5, min_f0=50,
                                    max_f0=1400, **kwargs):
    signal, fs = load_audio(filename, fs)

    # Preprocess the signal
    preprocessed_signal = preprocess_signal(signal, fs)

    # Frame the signal
    frames = np.array([preprocessed_signal[i:i + frame_length] for i in
                       range(0, len(preprocessed_signal) - frame_length, hop_length)])

    all_f0_estimates = []

    for frame in frames:
        f0_estimates = estimate_multiple_f0(frame, fs, num_pitches, min_f0, max_f0)
        all_f0_estimates.append(f0_estimates)

    return [list(set(frame_estimates)) for frame_estimates in all_f0_estimates]

def hz_to_piano_key(frequency):
    if frequency <= 0:
        return None

    key_number = 12 * math.log2(frequency / 440) + 49
    key_number = round(key_number)

    if 0 <= key_number <= 88:
        return key_number
    else:
        return None

# Example usage
if __name__ == "__main__":
    fs = 44100  # Sample rate
    f0_estimates = detect_pitch_kt('../../../audio/wlazl_kotek/wlazl_kotek.mp3', fs)
    print("Estimated F0 values for each frame:")
    for i, frame_estimates in enumerate(f0_estimates):
        frame_estimates = [hz_to_piano_key(f0) for f0 in frame_estimates]
        print(f"Frame {i + 1}: {set(frame_estimates)}")