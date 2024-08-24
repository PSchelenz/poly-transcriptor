import librosa
import numpy as np

from definitions import ROOT_DIR
import os

DATA_CONFIG = {
    "midi_resolution": 480,
    "save_to": os.path.join(ROOT_DIR, 'results/pitch'),
    "tracks": [
        {
            'name': 'wlazl_kotek',
            'dir': os.path.join(ROOT_DIR, 'audio/wlazl_kotek'),
            'audio': 'wlazl_kotek.mp3',
            'midi': 'wlazl_kotek.mid',
            'qpm': 100,
            'tolerance_s': 50,
            'scaling_correction': 35
        },
        {
            'name': 'lot_trzmiela',
            'dir': os.path.join(ROOT_DIR, 'audio/lot_trzmiela'),
            'audio': 'lot-trzmiela.mp3',
            'midi': 'lot-trzmiela.mid',
            'qpm': 140,
            'tolerance_s': 30,
            'scaling_correction': 55
        },
        {
            'name': 'a_kiedy_piano',
            'dir': os.path.join(ROOT_DIR, 'audio/a_kiedy'),
            'audio': 'a_kiedy_piano_v2.mp3',
            'midi': 'a_kiedy.mid',
            'qpm': 146,
            'tolerance_s': 30,
            'scaling_correction': 55
        },
        {
            'name': 'a_kiedy_guitars',
            'dir': os.path.join(ROOT_DIR, 'audio/a_kiedy'),
            'audio': 'a_kiedy_guitars.mp3',
            'midi': 'a_kiedy.mid',
            'qpm': 146,
            'tolerance_s': 30,
            'scaling_correction': 55
        },
        {
            'name': 'a_kiedy_violas',
            'dir': os.path.join(ROOT_DIR, 'audio/a_kiedy'),
            'audio': 'a_kiedy_violas.mp3',
            'midi': 'a_kiedy.mid',
            'qpm': 146,
            'tolerance_s': 30,
            'scaling_correction': 55
        },
        {
            'name': 'a_kiedy_all_instruments',
            'dir': os.path.join(ROOT_DIR, 'audio/a_kiedy'),
            'audio': 'a_kiedy_all_instruments.mp3',
            'midi': 'a_kiedy.mid',
            'qpm': 110,
            'tolerance_s': 40,
            'scaling_correction': 55
        },
        {
            'name': 'juice_mono',
            'dir': os.path.join(ROOT_DIR, 'audio/lizzy_juice'),
            'audio': 'juice-mono.mp3',
            'midi': 'juice-mono.mid',
            'qpm': 110,
            'tolerance_s': 80,
            'scaling_correction': 30
        },
        {
            'name': 'juice_harmony',
            'dir': os.path.join(ROOT_DIR, 'audio/lizzy_juice'),
            'audio': 'juice-harmony.mp3',
            'midi': 'juice-harmony.mid',
            'qpm': 110,
            'tolerance_s': 80,
            'scaling_correction': 30
        },
        {
            'name': 'juice_dissonance',
            'dir': os.path.join(ROOT_DIR, 'audio/lizzy_juice'),
            'audio': 'juice-dissonance.mp3',
            'midi': 'juice-dissonance.mid',
            'qpm': 110,
            'tolerance_s': 80,
            'scaling_correction': 30
        },
    ]
}

PITCH_DETECTOR_TO_FILENAME_MAPPERS = {
    # 'kt': 'karjalainen_tolonen',
    # 'pi2': 'pi2',
    'cbpdn': 'conv_bpdn',
    # 'siplca': 'si_plca'
}

PITCH_DETECTOR_TO_HUMAN_READABLE = {
    'kt': 'Karjalainen-Tolonen',
    'pi2': 'PI2',
    'cbpdn': 'ConvBPDN',
    'siplca': 'SI-PLCA',
}

tracks_qpm = {
    'wlazl_kotek': 100,
    'lot-trzmiela': 140,
    'blue-rondo': 86,
    'ipanema': 120,
    'juice': 110,
    'kolysanka': 60,
    'a_kiedy': 146,
}

def load_audio(filename, sr=44100):
    signal, fs = librosa.load(filename, sr=sr)

    return np.trim_zeros(signal), fs