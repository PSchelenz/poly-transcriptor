import librosa
import numpy as np
from sporco.admm import cbpdn
import pickle
import os
from scipy.signal import find_peaks

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octaves = ['0', '1', '2', '3', '4', '5', '6', '7', '8']


def detect_pitch_cbpdn(filename, sr=11025, window_size=256, hop_length=128, **kwargs):
    window_size_sec = window_size / sr

    # Load or create dictionary
    # try:
    #     with open('mapsdict.pkl', 'rb') as fid:
    #         D = pickle.load(fid)
    # except FileNotFoundError:
    D = []
    directory = '/home/piotr/Magisterka/code/audio/piano/remastered/notes'

    print('Loading piano notes...')
    for octave in octaves:
        for note in notes:
            # Skip notes that don't exist on a standard piano
            if (octave == '0' and note not in ['A', 'A#', 'B']) or (octave == '8' and note != 'C'):
                continue

            note_filename = os.path.join(directory, f"{note}{octave}.wav")
            y, sr = librosa.load(note_filename, sr=sr, duration=0.95)

            # Normalize the audio
            D.append(y / np.amax(y))

            print(f"Loaded {note}{octave}")

    D = np.asarray(D)

    D = D.T
    with open('mapsdict.pkl', 'wb') as fid:
        pickle.dump(D, fid)

    # Load the song
    song, sr = librosa.load(filename, sr=sr, mono=True)

    # Calculate duration based on number of samples and sample rate
    duration = len(song) / sr

    # CBPDN algorithm parameters
    lmbda = 0.001  # Reduced lambda for potentially sparser representation
    opt = cbpdn.ConvBPDN.Options({
        'Verbose': True,
        'MaxMainIter': 1000,  # Increased iterations
        'RelStopTol': 1e-4,  # Tighter tolerance
        'AuxVarObj': False,
        'AutoRho': {'Enabled': True}  # Enable automatic penalty parameter selection
    })

    # Run CBPDN algorithm
    print('Running CBPDN algorithm...')
    b = cbpdn.ConvBPDN(D, song, lmbda, opt, dimN=1)
    X = b.solve()
    X = X[:, 0, 0, :]

    # Process results
    Y = np.abs(X)  # Use absolute values instead of thresholding

    # Detect pitches
    pitches = []
    num_windows = int(duration / (window_size / sr))
    for i in range(num_windows):
        start = int(i * hop_length)
        end = start + window_size
        if end > Y.shape[0]:
            break
        frame = Y[start:end, :]
        frame_sum = np.sum(frame, axis=0)

        # Find peaks in the frame sum
        peaks, _ = find_peaks(frame_sum, height=np.max(frame_sum) * 0.1, distance=3)

        if len(peaks) > 0:
            pitches.append({
                'time': i * hop_length / sr,
                'notes': peaks + 21,  # Assuming 21 is your pitch offset
                'strengths': frame_sum[peaks]
            })

    return pitches