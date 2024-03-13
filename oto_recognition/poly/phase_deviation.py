import numpy as np
import librosa
import matplotlib.pyplot as plt

'''
Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler,
"On the use of phase and energy for musical onset detection in the
complex domain",
IEEE Signal Processing Letters, Volume 11, Number 6, 2004.
'''

def wrap_to_pi(phase):
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi

def _phase_deviation(phase):
    pd = np.zeros_like(phase)
    # instantaneous frequency is given by the first difference
    # ψ′(n, k) = ψ(n, k) − ψ(n − 1, k)
    # change in instantaneous frequency is given by the second order difference
    # ψ′′(n, k) = ψ′(n, k) − ψ′(n − 1, k)
    pd[2:] = phase[2:] - 2 * phase[1:-1] + phase[:-2]
    # map to the range -pi..pi
    return np.asarray(wrap_to_pi(pd))

def phase_deviation(phase):
    # absolute phase changes in instantaneous frequency
    pd = np.abs(_phase_deviation(phase))
    return np.asarray(np.mean(pd, axis=0))

def weighted_phase_deviation(spectrogram, phase):
    """
    Simon Dixon,
    "Onset Detection Revisited",
    Proceedings of the 9th International Conference on Digital Audio
    Effects (DAFx), 2006.
    """
    # make sure the spectrogram is not filtered before
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError('spectrogram and phase must be of same shape')
    # weighted_phase_deviation = spectrogram * phase_deviation
    wpd = np.abs(_phase_deviation(phase) * spectrogram)
    return np.asarray(np.mean(wpd, axis=0))

def normalized_weighted_phase_deviation(spectrogram, phase, epsilon=np.finfo(float).eps):
    """
    Simon Dixon,
    "Onset Detection Revisited",
    Proceedings of the 9th International Conference on Digital Audio
    Effects (DAFx), 2006.
    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    # normalize WPD by the sum of the spectrogram
    # (add a small epsilon so that we don't divide by 0)
    norm = np.add(np.mean(spectrogram, axis=0), epsilon)
    return np.asarray(weighted_phase_deviation(spectrogram, phase) / norm)

filename = '../../audio/midi_tracks/Canon_in_D.mp3'

y, sr = librosa.load(filename, duration=30)
stft = librosa.stft(y)
S = np.abs(stft)
phase = np.angle(stft)
pd = normalized_weighted_phase_deviation(S, phase)

onsets = librosa.util.peak_pick(pd, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=0)

# Convert frame indices to time
times = librosa.frames_to_time(onsets, sr=sr)

# Plotting
plt.figure(figsize=(14, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.vlines(times, -1, 1, color='r', alpha=0.9, label='Onsets')
plt.legend()
plt.title('Phase Deviation Onset Detection')
plt.show()