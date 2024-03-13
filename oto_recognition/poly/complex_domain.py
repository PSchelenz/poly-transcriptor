import numpy as np
import librosa
import matplotlib.pyplot as plt
import madmom


def _complex_domain(spectrogram):
    """
    Simon Dixon,
    "Onset Detection Revisited",
    Proceedings of the 9th International Conference on Digital Audio
    Effects (DAFx), 2006.
    """

    phase = spectrogram.stft.phase()
    # make sure the spectrogram is not filtered before
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError('spectrogram and phase must be of same shape')
    # expected spectrogram
    cd_target = np.zeros_like(phase)
    # assume constant phase change
    cd_target[1:] = 2 * phase[1:] - phase[:-1]
    # add magnitude
    cd_target = spectrogram * np.exp(1j * cd_target)
    # create complex spectrogram
    cd = spectrogram * np.exp(1j * phase)
    # subtract the target values
    cd[1:] -= cd_target[:-1]
    return np.asarray(cd)

def complex_domain(spectrogram):
    """
    Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler,
    "On the use of phase and energy for musical onset detection in the
    complex domain",
    IEEE Signal Processing Letters, Volume 11, Number 6, 2004.
    """
    # take the sum of the absolute changes
    return np.asarray(np.sum(np.abs(_complex_domain(spectrogram)), axis=1))

filename = '../../audio/midi_tracks/Canon_in_D.mp3'

stft = madmom.audio.stft.STFT(filename, num_channels=1, sample_rate=44100, frame_size=2048, hop_size=441, start=0, stop=30)

spec = madmom.audio.spectrogram.Spectrogram(stft)

cd = complex_domain(spec)

# Onset detection
onsets = madmom.features.onsets.peak_picking(cd, threshold=10, pre_max=1, post_max=1, pre_avg=1, post_avg=1)

print(len(onsets))

y, sr = librosa.load(filename, duration=30, sr=44100)

# Convert frame indices to time
times = librosa.frames_to_time(onsets, sr=sr, hop_length=441, n_fft=2048)

# Plotting
plt.figure(figsize=(14, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.vlines(times, -1, 1, color='r', alpha=0.9, label='Onsets')
plt.legend()
plt.title('Phase Deviation Onset Detection')
plt.show()