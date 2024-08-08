import numpy as np
from scipy.signal import find_peaks, butter, lfilter, hilbert
from scipy.fft import fft, ifft
import librosa
import matplotlib.pyplot as plt

from pitch_recognition.configurator import load_audio

def allpass_coefficients(lambda_):
    """Generate coefficients for a first-order allpass filter."""
    return [lambda_, 1], [1, lambda_]

def apply_allpass_filter(signal, lambda_):
    """Apply a first-order allpass filter to warp a signal."""
    b, a = allpass_coefficients(lambda_)
    return lfilter(b, a, signal)

def lpc_analysis(signal, order):
    """Compute LPC coefficients using librosa's lpc method."""
    return librosa.lpc(signal, order=order)


def inverse_warped_linear_prediction(signal, order=10, lambda_=0.5):
    # Inverse warp the signal
    warped_signal = apply_allpass_filter(signal, -lambda_)

    # Perform LPC on the warped signal
    lpc_coeffs = lpc_analysis(warped_signal, order)

    # Synthesize signal from LPC coefficients
    # Here, we use a very simple synthesis just for demonstration purposes
    lpc_synthesis = lfilter([1], lpc_coeffs, warped_signal)

    return lpc_synthesis


def split_signal(signal, fs):
    # Splitting signal into frequencies below and above 1 kHz
    b_low, a_low = butter(4, 1000/(fs/2), btype='low')
    b_high, a_high = butter(4, 1000/(fs/2), btype='high')
    low_freq_signal = lfilter(b_low, a_low, signal)
    high_freq_signal = lfilter(b_high, a_high, signal)
    return low_freq_signal, high_freq_signal

def generalized_acf(signal, alpha=0.67):
    # Applying the Generalized Autocorrelation Function
    spectrum = fft(signal)
    magnitude = np.abs(spectrum) ** alpha
    return ifft(magnitude).real


def sacf_enhancement(sacf):
    # plot sacf
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(sacf[:len(sacf) // 2])

    # Initialize the enhanced SACF (S'(r)) to be equal to the original SACF (S(r))
    enhanced_sacf = np.copy(sacf)

    # Start with the scaling factor m initialized to 2
    m = 2
    while m < 6:
        # Initialize an array for the time-scaled SACF Sm(r)
        Sm = np.zeros_like(sacf)

        # Perform linear interpolation to calculate Sm(r)
        for r in range(len(Sm)):
            d = r // m
            if d < len(sacf) - 1:
                Sm[r] = sacf[d] + ((r - m * d) / m) * (sacf[d + 1] - sacf[d])
            elif d == len(sacf) - 1:  # Handling the boundary case
                Sm[r] = sacf[d]

        # Update the enhanced SACF S'(r)
        enhanced_sacf = np.maximum(0, enhanced_sacf - np.maximum(0, Sm))

        # Increment m by 1
        m += 1


    # plot enhanced sacf
    # plt.subplot(2, 1, 2)
    # plt.plot(enhanced_sacf[:len(enhanced_sacf) // 2])
    # plt.title('SACF and Enhanced SACF')
    # plt.show()

    return enhanced_sacf

def find_f0s(sacf, fs):
    peaks, properties = find_peaks(sacf[:len(sacf) // 2], height=np.max(sacf)*0.4)
    f0s = fs / peaks
    amplitudes = properties['peak_heights']
    return f0s, amplitudes

# def select_dominant_frequency(frequencies, amplitudes):
#     if len(frequencies) == 0:
#         return None
#     sorted_indices = np.argsort(-amplitudes)
#     sorted_frequencies = frequencies[sorted_indices]
#     sorted_amplitudes = amplitudes[sorted_indices]
#
#     threshold = 0.8 * sorted_amplitudes[0]
#     significant_frequencies = sorted_frequencies[sorted_amplitudes > threshold]
#     return significant_frequencies if len(significant_frequencies) > 0 else None

def get_sacf(low_freq_acf, high_freq_acf, fs):
    sacf = low_freq_acf + high_freq_acf

    max_lag = int(fs / 1000)
    # Zero out the initial lags from 0 to max_lag in the SACF
    sacf[:max_lag] = 0
    return sacf

def estimate_f0s(signal, fs):
    inverse_warped_signal = inverse_warped_linear_prediction(signal)
    low_freq_signal, high_freq_signal = split_signal(inverse_warped_signal, fs)

    # plot low and high freq signal
    # plt.figure(figsize=(14, 8))
    # plt.subplot(4, 1, 1)
    # plt.plot(signal)
    # plt.title('Signal')
    # plt.subplot(4, 1, 2)
    # plt.plot(inverse_warped_signal)
    # plt.title('Inverse Warped Signal')
    # plt.subplot(4, 1, 3)
    # plt.plot(high_freq_signal)
    # plt.title('High Frequency Signal')
    # plt.subplot(4, 1, 4)
    # plt.plot(low_freq_signal)
    # plt.title('Low Frequency Signal')
    # plt.show()

    low_freq_acf = generalized_acf(low_freq_signal)
    high_freq_acf = generalized_acf(np.abs(high_freq_signal))  # Envelope of high-frequency channel

    sacf = get_sacf(low_freq_acf, high_freq_acf, fs)

    enhanced_sacf = sacf_enhancement(sacf)
    f0s, amplitudes = find_f0s(enhanced_sacf, fs)
    return f0s


def frame_signal(signal, frame_size, hop_length):
    # Calculate the total number of frames needed
    num_frames = 1 + int((len(signal) - frame_size) / hop_length)

    # Calculate padding length if the last frame exceeds the signal length
    padding_length = (num_frames * hop_length + frame_size) - len(signal)
    if padding_length > 0:
        # Pad signal with zeros at the end
        signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=(0, 0))

    # Create an array to hold the frames
    frames = np.zeros((num_frames, frame_size))

    # Hamming window
    window = np.hanning(frame_size)

    # Fill the frames array with windowed frames of the signal
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_size
        frames[i, :] = signal[start:end] * window

    return frames

def normalize_signal(x):
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return x  # Return the original if max is 0 to avoid division by zero
    return x / max_val

def _plot(x):
    plt.figure()
    plt.plot(x)
    plt.show()


def detect_pitch_kt(filename, sr=44100, hop_length=512, frame_size=2048, **kwargs):
    x = load_audio(filename, sr)[0]
    x = normalize_signal(x)

    frames = frame_signal(x, frame_size, hop_length)

    dominant_f0s = []

    for frame in frames:
        dominant_f0 = estimate_f0s(frame, sr)
        dominant_f0s.append(dominant_f0)

    return dominant_f0s