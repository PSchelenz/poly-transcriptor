from definitions import ROOT_DIR
import os

DATA_CONFIG = {
    "midi_resolution": 480,
    "save_to": os.path.join(ROOT_DIR, 'results/oto'),
    "tracks": [
        {
            'name': 'wlazl_kotek',
            'dir': os.path.join(ROOT_DIR, 'audio/wlazl_kotek'),
            'audio': 'wlazl_kotek.mp3',
            'midi': 'wlazl_kotek.mid',
            'qpm': 100,
            'tolerance_s': 50,
            'scaling_correction': 30,
        },
        {
            'name': 'lot_trzmiela',
            'dir': os.path.join(ROOT_DIR, 'audio/lot_trzmiela'),
            'audio': 'lot-trzmiela.mp3',
            'midi': 'lot-trzmiela.mid',
            'qpm': 140,
            'tolerance_s': 30,
            'scaling_correction': 50,
        },
        {
            'name': 'kolysanka_guitars',
            'dir': os.path.join(ROOT_DIR, 'audio/kolysanka'),
            'audio': 'kolysanka_guitar.mp3',
            'midi': 'kolysanka.mid',
            'qpm': 60,
            'tolerance_s': 80,
            'scaling_correction': 30,
        },
        {
            'name': 'kolysanka_strings',
            'dir': os.path.join(ROOT_DIR, 'audio/kolysanka'),
            'audio': 'kolysanka_strings.mp3',
            'midi': 'kolysanka.mid',
            'qpm': 60,
            'tolerance_s': 80,
            'scaling_correction': 30,
        },
        {
            'name': 'kolysanka_woodwinds',
            'dir': os.path.join(ROOT_DIR, 'audio/kolysanka'),
            'audio': 'kolysanka_woodwind.mp3',
            'midi': 'kolysanka.mid',
            'qpm': 60,
            'tolerance_s': 80,
            'scaling_correction': 30,
        }
    ]
}

ONSET_DETECTOR_TO_FILENAME_MAPPERS = {
    # 'sd': 'spectral_diff',
    # 'sf': 'spectral_flux',
    # 'hfc': 'hfc',
    # 'pd': 'phase_deviation',
    'rnn': 'rnn'
}

ONSET_DETECTOR_TO_HUMAN_READABLE = {
    'sd': 'Spectral Difference',
    'sf': 'Spectral Flux',
    'hfc': 'High Frequency Content',
    'pd': 'Phase Deviation',
    'rnn': 'Recurrent Neural Network'
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