import os.path

import numpy as np
import librosa
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, lil_matrix
from scipy.signal import filtfilt, medfilt, butter
from scipy.signal.windows import blackmanharris, hann, blackman
from scipy.fft import fft
import warnings
import matplotlib.pyplot as plt
import pathlib

from pitch_recognition.configurator import load_audio, ROOT_DIR

globalY = None
globalPA = None
globalW = None
notes_count = 88
min_freq = 27.5
characteristics = 264


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def _plot(arr):
    for note in range(0, 88):
        plotting = arr[2, note, :, 0]
        plt.plot(plotting)
        plt.show()


def transcription(filename, sr, frame_size, hop_length, iter, S, sz, su, sh, model_path=os.path.join(ROOT_DIR, 'pitch_recognition/poly/si_plca/shiftedW'), min_draw_note = 1, max_draw_note = 60, draw_threshold = 0.01, draw_notes_longer_than = 8, notes_count = 88, instrument = 'piano'):
    global globalY, globalPA, globalW, characteristics

    # shiftedW = sio.loadmat(model_path)['shiftedW']
    shiftedW = np.load(os.path.join(ROOT_DIR, f'audio/piano/remastered_v2/notes/{instrument}_template_{S}r.npy'))
    pitchActivity = np.array([[1] * S, [notes_count] * S]).T
    W = np.transpose(shiftedW, (2, 3, 1, 0))
    W = W[:, :S, :, :]

    characteristics = W.shape[0]

    samples = load_audio(filename, sr)[0]

    Y = np.abs(librosa.cqt(samples, sr=sr, n_bins=(notes_count + notes_count//2) * 4,
                       fmin=27.5,
                       bins_per_octave=12 * 4)).T

    # intCQT = compute_cqt(filename, sr)
    # X = intCQT[:, np.round(np.arange(0, intCQT.shape[1], 7.1129)).astype(int)].T
    # print('Calc noise')
    # noiseLevel1 = medfilt(X.T, kernel_size=[41, 1])
    # print('Calc noise2')
    # noiseLevel2 = medfilt(np.minimum(X.T, noiseLevel1), kernel_size=[41, 1])
    # X = np.maximum(X - noiseLevel2.T, 0)
    Y = Y[::4, :]  # 40ms step

    globalY = Y
    globalPA = pitchActivity
    globalW = W

    # w, h, z, u, xa = mssiplca_fast(Y.T, 88, S, 5, iter, sh, sz, su, W, None, None, None, 1, pitchActivity)

    # pianoRoll = z

    # pianoRoll = filter_notes(pianoRoll, draw_threshold, draw_notes_longer_than)

    # plt.pcolormesh(np.arange(pianoRoll.shape[1]), np.arange(max_draw_note - min_draw_note + 1),
    #                pianoRoll[min_draw_note - 1 : max_draw_note],
    #                shading='auto', cmap='binary')

    # return pianoRoll[min_draw_note - 1 : max_draw_note]

def compute_cqt(filename, sr):
    # Load audio file
    y, fs = load_audio(filename, sr)

    # If stereo, convert to mono by averaging the channels
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Resample to 44100 Hz if necessary
    if fs != sr:
        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
        fs = sr

    # Compute CQT
    Xcqt = cqt(y, 27.5, fs / 3, 60, fs, q=0.8, atomHopFactor=0.3, thresh=0.0005, win='hann')

    # Obtain absolute CQT (assuming a getCQT function exists)
    absCQT = getCQT(Xcqt, 'all', 'all')

    # Crop CQT to useful time regions
    emptyHops = Xcqt['intParams']['firstcenter'] / Xcqt['intParams']['atomHOP']
    maxDrop = emptyHops * 2 ** (Xcqt['octaveNr'] - 1) - emptyHops
    droppedSamples = (maxDrop - 1) * Xcqt['intParams']['atomHOP'] + Xcqt['intParams']['firstcenter']
    outputTimeVec = (np.arange(absCQT.shape[1]) + 1) * Xcqt['intParams']['atomHOP'] - Xcqt['intParams'][
        'preZeros'] + droppedSamples

    lowerLim = np.argmax(outputTimeVec > 0)
    upperLim = np.argmax(outputTimeVec > len(y))

    if upperLim == 0:
        upperLim = len(outputTimeVec)

    # Final cropped CQT
    intCQT = absCQT[:characteristics, lowerLim:upperLim]

    return intCQT


def buffer(x, n, overlap, nodelay=False):
    if nodelay:
        start_index = 0
    else:
        start_index = n - overlap

    # Calculate the number of columns in the buffer matrix
    num_cols = int(np.ceil((x.size - n) / (n - overlap))) + 1
    # Initialize the buffer matrix with zeros
    buff = np.zeros((n, num_cols))

    for i in range(num_cols):
        # Column-wise assignment
        length = np.min((start_index + i * (n - overlap) + n, x.size)) - (start_index + i * (n - overlap))

        buff[:, i] = np.pad(x[start_index + i * (n - overlap): start_index + i * (n - overlap) + n], (0, n - length), 'constant', constant_values=0)
    return buff

def cqt(x, fmin, fmax, bins, fs, **kwargs):
    # Default parameter values
    q = kwargs.get('q', 1)
    atomHopFactor = kwargs.get('atomHopFactor', 0.25)
    thresh = kwargs.get('thresh', 0.0005)
    winFlag = kwargs.get('win', 'sqrt_blackmanharris')

    # Additional parameters, use kwargs.get to provide default values if not specified
    B = kwargs.get('coeffB', None)
    A = kwargs.get('coeffA', None)
    cqtKernel = kwargs.get('kernel', None)

    if x.ndim > 1 and x.shape[0] > 1:
        raise ValueError('cqt requires one-dimensional input!')
    if x.ndim > 1:
        x = x.ravel()  # Convert to column vector if necessary

    octaveNr = np.ceil(np.log2(fmax / fmin))
    xlen_init = len(x)

    # Design anti-aliasing filter if B and A are not provided
    if B is None or A is None:
        LPorder = 6  # Order of the anti-aliasing filter
        cutoff = 0.5
        B, A = butter(LPorder, cutoff, 'low')

    # Generate CQT kernel if not provided
    if cqtKernel is None:
        # Placeholder for genCQTkernel function, you'll need to implement or replace this
        cqtKernel = genCQTkernel(fmax, bins, fs, q=q, atomHopFactor=atomHopFactor, thresh=thresh, winFlag=winFlag)

    # Main CQT processing
    cellCQT = []
    maxBlock = cqtKernel['fftLEN'] * 2 ** (octaveNr - 1)
    suffixZeros = int(maxBlock)
    prefixZeros = int(maxBlock)
    x = np.concatenate((np.zeros(prefixZeros), x, np.zeros(suffixZeros))) # TODO: not sure which one is correct

    OVRLP = cqtKernel['fftLEN'] - cqtKernel['fftHOP']
    K = np.conj(cqtKernel['fKernel'].T)
    for i in range(int(octaveNr)):
        # xx = np.lib.stride_tricks.sliding_window_view(x, int(cqtKernel['fftLEN']))[::int(OVRLP) + 1]
        xx = buffer(x, cqtKernel['fftLEN'], int(OVRLP), nodelay=True)
        XX = np.fft.fft(xx, axis=0)
        cellCQT.append(K @ XX)
        if i != octaveNr - 1:
            x = filtfilt(B, A, x)
            x = x[::2]  # Decimate by 2

    spCQT = cell2sparse(cellCQT, octaveNr, bins, cqtKernel['firstcenter'], cqtKernel['atomHOP'], cqtKernel['atomNr'])

    intParams = {'sufZeros': suffixZeros, 'preZeros': prefixZeros, 'xlen_init': xlen_init,
                 'fftLEN': cqtKernel['fftLEN'],
                 'fftHOP': cqtKernel['fftHOP'], 'q': q, 'filtCoeffA': A, 'filtCoeffB': B,
                 'firstcenter': cqtKernel['firstcenter'],
                 'atomHOP': cqtKernel['atomHOP'], 'atomNr': cqtKernel['atomNr'], 'Nk_max': cqtKernel['Nk_max'],
                 'Q': cqtKernel['Q'], 'rast': 0}

    Xcqt = {'spCQT': spCQT, 'fKernel': cqtKernel['fKernel'], 'fmax': fmax, 'fmin': fmin * 2 ** (1 / bins),
            'octaveNr': octaveNr, 'bins': bins, 'intParams': intParams}

    return Xcqt

def genCQTkernel(fmax, bins, fs, q=1, atomHopFactor=0.25, thresh=0.0005, winFlag='sqrt_blackmanharris', perfRast=0):
    fmin = (fmax / 2) * 2 ** (1 / bins)
    Q = 1 / (2 ** (1 / bins) - 1)
    Q *= q
    Nk_max = Q * fs / fmin
    Nk_max = round(Nk_max)

    Nk_min = round(Q * fs / (fmin * 2 ** ((bins - 1) / bins)))
    atomHOP = round(Nk_min * atomHopFactor)
    first_center = np.ceil(Nk_max / 2)
    first_center = atomHOP * np.ceil(first_center / atomHOP)
    FFTLen = nextpow2(first_center + np.ceil(Nk_max / 2))

    if perfRast:
        winNr = (FFTLen - np.ceil(Nk_max / 2) - first_center) // atomHOP
        if winNr == 0:
            FFTLen *= 2
            winNr = (FFTLen - np.ceil(Nk_max / 2) - first_center) // atomHOP
    else:
        winNr = ((FFTLen - np.ceil(Nk_max / 2) - first_center) // atomHOP) + 1

    last_center = first_center + (winNr - 1) * atomHOP
    fftHOP = (last_center + atomHOP) - first_center
    fftOLP = (FFTLen - fftHOP / FFTLen) * 100

    sparKernel = []
    for k in range(1, bins + 1):
        Nk = round(Q * fs / (fmin * 2 ** ((k - 1) / bins)))
        if winFlag == 'sqrt_blackmanharris':
            winFct = np.sqrt(blackmanharris(Nk))
        elif winFlag == 'blackmanharris':
            winFct = blackmanharris(Nk)
        elif winFlag == 'sqrt_hann':
            winFct = np.sqrt(hann(Nk, sym=False))
        elif winFlag == 'hann':
            winFct = hann(Nk, sym=False)
        elif winFlag == 'sqrt_blackman':
            winFct = np.sqrt(blackman(Nk, sym=False))
        elif winFlag == 'blackman':
            winFct = blackman(Nk, sym=False)
        else:
            winFct = np.sqrt(blackmanharris(Nk))
            if k == 1:
                warnings.warn('Non-existing window function. Default window is used!', category=UserWarning)

        fk = fmin * 2 ** ((k - 1) / bins)
        tempKernelBin = (winFct / Nk) * np.exp(2 * np.pi * 1j * fk * np.arange(Nk) / fs)
        atomOffset = first_center - np.ceil(Nk / 2)

        for i in range(int(winNr)):
            shift = int(atomOffset + (i * atomHOP))
            tempKernel = np.zeros(FFTLen, dtype=complex)
            tempKernel[shift:Nk + shift] = tempKernelBin
            specKernel = fft(tempKernel)
            specKernel[np.abs(specKernel) <= thresh] = 0
            sparKernel.append(specKernel)

    sparKernel = np.array(sparKernel).T / FFTLen
    wx1 = np.argmax(sparKernel[:, 0])
    wx2 = np.argmax(sparKernel[:, -1])
    wK = sparKernel[wx1:wx2+1, :]
    wK = np.diag(wK @ wK.T)
    wK = wK[int(np.round(1 / q)):(-int(np.round(1 / q)) - 2)]
    weight = 1. / np.mean(np.abs(wK))
    weight *= fftHOP / FFTLen
    weight = np.sqrt(weight)
    sparKernel = weight * sparKernel

    cqtKernel = {'fKernel': sparKernel, 'fftLEN': FFTLen, 'fftHOP': fftHOP, 'fftOverlap': fftOLP, 'perfRast': perfRast,
                 'bins': bins, 'firstcenter': first_center, 'atomHOP': atomHOP, 'atomNr': winNr, 'Nk_max': Nk_max,
                 'Q': Q, 'fmin': fmin}
    return cqtKernel

def cell2sparse(Xcq, octaves, bins, firstcenter, atomHOP, atomNr):
    emptyHops = firstcenter / atomHOP
    drops = emptyHops * 2 ** (np.arange(octaves) + 1 - 1) - emptyHops
    drops = drops[::-1]
    atomNr = int(atomNr)
    len_max = np.max(((atomNr * np.array([x.shape[1] for x in Xcq]) - drops) * 2 ** np.arange(octaves)))

    spCQT_list = []
    for i in range(int(octaves), 0, -1):
        drop = int(emptyHops * 2 ** (octaves - i) - emptyHops)
        X = Xcq[i - 1]
        if atomNr > 1:
            Xoct = np.zeros((bins, atomNr * X.shape[1] - drop))
            for u in range(bins):
                octX_bin = X[(u * atomNr):((u + 1) * atomNr), :]
                Xcont = octX_bin.reshape(-1)
                Xoct[u, :] = Xcont[drop:]
            X = Xoct
        else:
            X = X[:, drop:]

        X = np.repeat(X, 2 ** (i - 1), axis=1)
        if X.shape[1] < len_max:
            X = np.hstack((X, np.zeros((bins, int(len_max - X.shape[1])))))

        spCQT_list.append(X)

    spCQT = np.vstack(spCQT_list)
    spCQT_sparse = csr_matrix(spCQT)
    return spCQT_sparse

def getCQT(Xcqt, fSlice, tSlice, iFlag='linear'):
    # Handle dynamic type inputs for fSlice and tSlice
    if isinstance(fSlice, str):
        fSlice = np.arange(1, Xcqt['bins'] * Xcqt['octaveNr'] + 1)
    if isinstance(tSlice, str):
        first_row = Xcqt['spCQT'][0, :]
        if first_row.nnz > 0:
            lastEnt = first_row.indices[-1] + 1
        else:
            lastEnt = 0

        tSlice = np.arange(1, lastEnt + 1)

    # Initialize intCQT with zeros
    intCQT = np.zeros((len(fSlice), len(tSlice)))

    # Extract values from Xcqt
    bins = Xcqt['bins']
    spCQT = Xcqt['spCQT'].T  # Transposed to match MATLAB's behavior
    octaveNr = Xcqt['octaveNr']

    # Main loop for interpolation
    for k, f in enumerate(fSlice):
        oct = octaveNr - np.floor((f - 0.1) / bins)
        stepVec = np.arange(1, spCQT.shape[0] + 1, 2 ** (oct - 1))
        Xbin = np.abs(spCQT[stepVec.astype(int) - 1, int(f) - 1].toarray().flatten())
        interp_func = interp1d(stepVec, Xbin, kind=iFlag, fill_value='extrapolate')
        intCQT[k, :] = interp_func(tSlice)

    return intCQT

def mssiplca_fast(x, K, R, F, iter=100, sh=1.0, sz=1.0, su=1.0, w=None, h=None, z=None, u=None, pl=False, pa=None):
    # Get sizes
    M, N = x.shape
    sumx = np.sum(x, axis=0)

    # Initialize
    if w is None:
        w = np.random.rand(M, R, K, F)
    for r in range(R):  # normalize w
        for k in range(K):
            for f in range(F):
                w[:, r, k, f] /= (np.sum(w[:, r, k, f]) + np.finfo(float).eps)

    if z is None:
        z = np.random.rand(K, N)

    n = np.arange(N)
    z[:, n] = np.tile(sumx[n], (K, 1)) * (z[:, n] / np.tile(np.sum(z[:, n], axis=0), (K, 1)))

    if u is None:
        u = np.ones((R, K, N))
    for k in range(K):
        for r in range(R):
            if pa[r, 0] <= k <= pa[r, 1]:
                u[r, k, :] = np.ones((1, N))
    for k in range(K):
        for n in range(N):
            u[:, k, n] /= np.sum(u[:, k, n])

    if h is None:
        h = np.random.rand(F, K, N)
    for k in range(K):
        for n in range(N):
            h[:, k, n] /= np.sum(h[:, k, n])

    # Initialize update parameters
    w_reshaped = w.reshape(M, R * K * F)
    sumx = np.diag(sumx)

    # plt.figure(figsize=(20, 16))

    # Iterate
    for it in range(iter):
        print(f'Iteration: {it + 1}')
        # E-step
        uz = u * np.transpose(np.tile(z[:, :, np.newaxis], (1, 1, R)), (2, 0, 1))
        uz_big = np.transpose(np.tile(uz[:, :, :, np.newaxis], (1, 1, 1, F)), (2, 0, 1, 3))
        h_big = np.transpose(np.tile(h[:, :, :, np.newaxis], (1, 1, 1, R)), (2, 3, 1, 0))
        temp_uzh = uz_big * h_big
        temp_uzh_reshaped = temp_uzh.reshape((N, R * K * F))
        xa = w_reshaped @ temp_uzh_reshaped.T
        Q = x / xa
        # M-step (update h, z, u)
        WQ = Q.T @ w_reshaped
        WQUZH = WQ * temp_uzh_reshaped
        WQUZH = WQUZH.reshape((N, R, K, F))
        z = np.sum(np.sum(WQUZH, axis=1, keepdims=True), axis=3).squeeze().T ** sz
        h = np.transpose(np.sum(WQUZH, axis=1).squeeze(), (2, 1, 0)) ** sh
        u = np.power(np.transpose(np.sum(WQUZH, axis=3), (1, 2, 0)), su)
        # Normalize h, z, u
        z = np.dot(z / (np.sum(z, axis=0, keepdims=True) + np.finfo(float).eps), sumx)
        u_resh = u.reshape((R, K * N))
        u_resh /= (np.tile(np.sum(u_resh, axis=0, keepdims=True), (R, 1)) + np.finfo(float).eps)
        u = u_resh.reshape((R, K, N))
        h_resh = h.reshape((F, K * N))
        h_resh /= (np.tile(np.sum(h_resh, axis=0, keepdims=True), (F, 1)) + np.finfo(float).eps)
        h = h_resh.reshape((F, K, N))


    # if pl:
        # plt.subplot(3, 1, 1)
        # plt.imshow(x, aspect='auto', origin='lower')
        # plt.title(f'Git gud')
        # plt.subplot(3, 1, 2)
        # plt.imshow(xa, aspect='auto', origin='lower')
        # plt.subplot(3, 1, 3)
        # plt.imshow(z, aspect='auto', origin='lower')

    # plt.pcolormesh(np.arange(z.shape[1]), np.arange(z.shape[0]), z, shading='auto')
        # plt.show()

    return w, h, z, u, xa


#my code
def filter_notes(pianoRoll, threshold = 0.01, count_notes_lt = 8):
    # normalize notes
    normalized_pianoRoll = pianoRoll / np.max(pianoRoll)
    normalized_pianoRoll[normalized_pianoRoll < threshold] = 0
    normalized_pianoRoll[normalized_pianoRoll >= threshold] = 1

    # remove notes that are too short
    for note_index in range(normalized_pianoRoll.shape[0]):
        active_frames = np.diff(np.r_[0, normalized_pianoRoll[note_index, :], 0])
        starts = np.where(active_frames > 0)[0]
        ends = np.where(active_frames < 0)[0]
        for start, end in zip(starts, ends):
            if (end - start) < count_notes_lt:
                normalized_pianoRoll[note_index, start:end] = 0

    return normalized_pianoRoll

def detect_pitch_siplca(filename, sr = 44100, hop_length=256, frame_size=2048, **kwargs):
    global notes_count, min_freq

    R = 7
    notes_count = 46
    min_freq = 130
    instrument = 'viola'

    transcription(filename, sr, frame_size, hop_length, 50, R, 1.18, 1.15, 1, notes_count=notes_count, instrument=instrument)
    w, h, z, u, xa = mssiplca_fast(globalY.T, notes_count, R, 4, 50, 1.1, 1.2, 2, globalW, None, None, None, 1, globalPA)
    pianoRoll = filter_notes(z, 0.05, 1)

    #kotek 1.1 1.2 2 | 0.05 1
    #trzmiel 1.1 1.2 2 | 0.05 1
    #a_kiedy_piano 1.1 1.2 2 | 0.001 1
    #a_kiedy_guitar 1.1 1.2 2 | 0.01 1
    #a_kiedy_viola 1.1 1.2 2 | 0.01 1
    #a_kiedy_all 1.1 1.2 2 | 0.05 1
    #juice-mono 1.1, 1.2, 2 | 0.05 1
    #juice-harmony 1.1 1.1 2 | 0.001 1
    #juice-dissonance 1.1 1.1 2 | 0.001 1

    return pianoRoll

if __name__ == '__main__':
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.title.set_text('SH = 1.0, SZ = 1.2, SU = 1.2')
    transcription('../../../audio/midi_tracks/Canon_in_D.mp3', 50, 3, 1.18, 1.15, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax2.title.set_text('SH = 0.9, SZ = 1.2, SU = 1.2')
    w, h, z, u, xa = mssiplca_fast(globalY.T, 88, 3, 5, 50, 0.9, 1.18, 1.2, globalW, None, None, None, 1, globalPA)
    ax3 = plt.subplot(2, 2, 3)
    ax3.title.set_text('SH = 0.8, SZ = 1.2, SU = 1.2')
    w, h, z, u, xa = mssiplca_fast(globalY.T, 88, 3, 5, 50, 0.8, 1.18, 1.10, globalW, None, None, None, 1, globalPA)
    ax4 = plt.subplot(2, 2, 4)
    ax4.title.set_text('SH = 0.6, SZ = 1.2, SU = 1.2')
    w, h, z, u, xa = mssiplca_fast(globalY.T, 88, 3, 5, 50, 0.6, 1.18, 1.05, globalW, None, None, None, 1, globalPA)
    plt.show()
