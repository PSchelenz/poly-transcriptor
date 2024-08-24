from copy import deepcopy

import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from sporco.admm import cbpdn
import pickle

notes = ['C', 'Cis', 'D', 'Dis', 'E', 'F', 'Fis', 'G', 'Gis', 'A', 'Ais', 'H']
octaves = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

def detect_pitch_cbpdn(filename, sr=11025, hop_length=128, window_size=256, **kwargs):
    window_size_ms = window_size/sr

    D = []
    directory = '/home/piotr/Magisterka/code/audio/piano/full_notes/'
    dictname = 'mapsdict'

    print('Mapping piano notes...')
    for octave in octaves:
        for note in notes:
            if (octave == '0' and note not in ['A', 'Ais', 'H']) or (octave == '8' and note != 'C'):
                continue

            filename = directory + note + octave + '.wav'
            y, sr = librosa.load(filename, sr=sr, duration=1)

            D.append(y / np.amax(y))

    D = np.array(D)

    D = D.T

    with open(dictname + '.pkl', 'wb') as fid:
        pickle.dump(D, fid)

    with open(dictname + '.pkl', 'rb') as fid:
        D = pickle.load(fid)

    lmbda = 0.005

    dimN = 1

    opt = cbpdn.ConvBPDN.Options({'Verbose': True,
                                  'MaxMainIter': 500,
                                  'HighMemSolve': False,
                                  'LinSolveCheck': False,
                                  'RelStopTol': 1e-3,
                                  'AuxVarObj': False,
                                  })

    song, sr = librosa.load(filename, sr=sr)

    b = cbpdn.ConvBPDN(D, song, lmbda, opt, dimN=dimN)

    print('Solving ConvBPDN...')
    X = b.solve()

    X = X[:, 0, 0, :]

    results_name = 'maps_results_' + 'esp6' + '.pkl'

    if 'save_pickle' in kwargs:
        with open(results_name, 'wb') as fid:
            pickle.dump(X, fid)

    with open(results_name,'rb') as Y:
        Y1 = pickle.load(Y)

    Y = deepcopy(Y1)

    lmaxes = np.asarray([])

    for row in Y:
        relex = argrelextrema(row,np.greater)[0]
        lmaxes = np.concatenate((lmaxes,row[relex]))

    p75,p25 = np.percentile(lmaxes, [75,25]) # changed from 75,25

    threshold = p75 + 8*(p75-p25)

    lowvals = Y < threshold
    Y[lowvals] = 0

    start = 0
    l = int(11025 * window_size_ms)
    end = l
    window = 0

    # 2D array that represents onsets
    frames = int(len(song) / window_size)
    onsets = np.zeros((88, frames))
    # highest and lowest note (for plotting purposes)
    highnote = 0
    lownote = 100
    total = 0
    correct = 0
    poped = 0

    for i in range(frames):
        row = Y[start:end, :]
        # print(row.shape)
        sumrow = np.sum(row, axis=0)
        relex = argrelextrema(sumrow, np.greater)[0]
        if (len(relex) != 0):
            # time index that the note occurs
            high = np.amax(relex)
            low = np.amin(relex)
            if high > highnote:
                highnote = high
            if low < lownote:
                lownote = low
            tindex = window
            onsets[relex, tindex] = 1
            t = window_size_ms * window

        start = end
        end += l
        window += 1

    # remove temporal information
    for i in range(onsets.shape[1] - 1):
        curr = onsets[:, i]
        nxt = onsets[:, i + 1]
        # take differnce of two frames
        chnge = nxt - curr
        # where change = 0, we want to set next to 0
        indices = chnge < 1
        onsets[indices, i + 1] = 0

    # transform data so we can do a scatterplot
    x1 = []
    y1 = []

    for i in range(onsets.shape[1]):
        col1 = onsets[:, i]
        # get indices where note is playing
        counter = 0
        for c1 in col1:
            if c1:
                x1.append(i)
                y1.append(counter)
            counter += 1

    # fig, ax = plt.subplots(figsize=(40, 20))
    #
    # plt.legend(loc = "upper right",prop={'size':30})
    # ax.plot(np.asarray(x1)/(1. / window_size),y1,'bo',alpha=0.5,label = 'transcription', markersize=10)
    # major_ticks = np.arange(lownote,highnote+3)
    # ax.set_yticks(major_ticks)
    # ax.set_xticks(np.arange(0,31,5))
    # ax.grid(True)
    # ax.tick_params(axis='both', which='major', labelsize=8)
    # plt.ylim([23,highnote+3])
    # plt.title('Note Onsets for Transcription vs. Ground Truth, ' + 'ESP2',fontsize=20)
    # plt.xlabel('Time (s)',fontsize='30')
    # plt.ylabel('Note Index',fontsize="30")
    # plt.show()

    return onsets