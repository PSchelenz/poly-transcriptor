import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter

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

def normalize_signal(x):
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return x  # Return the original if max is 0 to avoid division by zero
    return x / max_val

# Example usage
lambda_value = 0.5  # Warping factor
order = 16  # Order of LPC

fs = 44100  # Sampling rate
 # Example signal
x = librosa.load("audio/midi_tracks/Canon_in_D.mp3", sr=fs, duration=1)[0]
signal = normalize_signal(x)

# Warp the signal
warped_signal = apply_allpass_filter(signal[4096:6192], lambda_value)
inverse_warped_signal = apply_allpass_filter(signal[4096:6192], -lambda_value)

# Perform LPC on the warped signal
lpc_coeffs = lpc_analysis(warped_signal, order)
inverse_lpc_coeffs = lpc_analysis(inverse_warped_signal, order)

# Synthesize signal from LPC coefficients
# Here, we use a very simple synthesis just for demonstration purposes
lpc_synthesis = lfilter([1], lpc_coeffs, warped_signal)
inverse_lpc_synthesis = lfilter([1], inverse_lpc_coeffs, inverse_warped_signal)

plt.figure(figsize=(18, 8))
plt.subplot(4, 1, 1)
plt.plot(signal[4096:6192])
plt.title('Original Signal')
plt.subplot(4, 1, 2)
plt.plot(warped_signal)
plt.title('Warped Signal')
plt.subplot(4, 1, 3)
plt.plot(lpc_synthesis)
plt.title('Signal Synthesized from LPC Coefficients of Warped Signal')
plt.subplot(4, 1, 4)
plt.plot(lpc_synthesis)
plt.title('Signal Synthesized from LPC Coefficients of Inverse Warped Signal')
plt.tight_layout()
plt.show()