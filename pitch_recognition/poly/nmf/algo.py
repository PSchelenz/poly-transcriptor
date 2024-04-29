import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

def _plot(signal):
    plt.figure()
    plt.plot(signal)
    plt.show()

# Load an audio file
audio_path = '../../../audio/midi_tracks/Canon_in_D.mp3'
y, sr = librosa.load(audio_path, sr=None, offset=0.2, duration=30)

# Define segment length and hop size
segment_length = int(sr * 0.1)  # e.g., 100 ms segments
hop_length = int(segment_length * 0.5)  # 50% overlap

for start in range(0, len(y) - segment_length, hop_length):
    segment = y[start:start + segment_length]

    _plot(segment)

    # Create a spectrogram
    S = np.abs(librosa.stft(segment))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    # Apply NMF to the spectrogram
    n_components = 10  # Number of components to extract
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    W = model.fit_transform(S)
    H = model.components_

    # Plot the results
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    img1 = librosa.display.specshow(S_dB, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set_title('Original Spectrogram')
    fig.colorbar(img1)

    for i, component in enumerate(W.T):
        ax[1].plot(component, label=f'Component {i+1}')
    ax[1].set_title('NMF Components')
    ax[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Frequency analysis on the NMF components
    frequencies = np.linspace(0, sr/2, len(W[:, 0]))
    for i, component in enumerate(H):
        max_idx = np.argmax(component)
        fundamental_freq = frequencies[max_idx]
        print(f"Estimated fundamental frequency for Component {i+1}: {fundamental_freq:.2f} Hz")