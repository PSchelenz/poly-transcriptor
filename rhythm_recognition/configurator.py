from definitions import ROOT_DIR
import os

DATA_CONFIG = {
    "midi_resolution": 480,
    "save_to": os.path.join(ROOT_DIR, 'results/rhythm'),
    "tracks": [
        {
            'name': 'wlazl_kotek',
            'dir': os.path.join(ROOT_DIR, 'audio/wlazl_kotek'),
            'audio': 'wlazl_kotek.mp3',
            'midi': 'wlazl_kotek.mid',
            'qpm': 100,
            'tolerance_s': 50,
            'scaling_correction': 35,
            'correct_beats': [i * 480 for i in range(17)]
        },
        {
            'name': 'lot_trzmiela',
            'dir': os.path.join(ROOT_DIR, 'audio/lot_trzmiela'),
            'audio': 'lot-trzmiela.mp3',
            'midi': 'lot-trzmiela.mid',
            'qpm': 140,
            'tolerance_s': 30,
            'scaling_correction': 55,
            'correct_beats': [i * 480 for i in range(46)]
        },
        {
            'name': 'kolysanka_guitars',
            'dir': os.path.join(ROOT_DIR, 'audio/kolysanka'),
            'audio': 'kolysanka_guitar.mp3',
            'midi': 'kolysanka.mid',
            'qpm': 60,
            'tolerance_s': 80,
            'scaling_correction': 30,
            'correct_beats': [i * 240 for i in range(48)]
        },
        {
            'name': 'kolysanka_strings',
            'dir': os.path.join(ROOT_DIR, 'audio/kolysanka'),
            'audio': 'kolysanka_strings.mp3',
            'midi': 'kolysanka.mid',
            'qpm': 60,
            'tolerance_s': 80,
            'scaling_correction': 30,
            'correct_beats': [i * 240 for i in range(48)]
        },
        {
            'name': 'kolysanka_woodwinds',
            'dir': os.path.join(ROOT_DIR, 'audio/kolysanka'),
            'audio': 'kolysanka_woodwind.mp3',
            'midi': 'kolysanka.mid',
            'qpm': 60,
            'tolerance_s': 80,
            'scaling_correction': 30,
            'correct_beats': [i * 240 for i in range(48)]
        },
        {
            'name': 'girl_from_ipanema',
            'dir': os.path.join(ROOT_DIR, 'audio/girl_from_ipanema'),
            'audio': 'ipanema-piano.mp3',
            'midi': 'ipanema-piano.mid',
            'qpm': 120,
            'tolerance_s': 40,
            'scaling_correction': 40,
            'correct_beats': [i * 960 for i in range(16)]
        },
        {
            'name': 'blue_rondo',
            'dir': os.path.join(ROOT_DIR, 'audio/blue_rondo'),
            'audio': 'blue-rondo.mp3',
            'midi': 'blue-rondo.mid',
            'qpm': 189,
            'tolerance_s': 40,
            'scaling_correction': 70,
            'correct_beats': [i * 240 for i in range(72)]
        }
    ]
}

BEAT_DETECTOR_TO_FILENAME_MAPPERS = {
    'dp': 'dynamic_programming',
    'dbn': 'dynamic_bayesian_network',
    'crf': 'conditional_random_fields',
}

BEAT_DETECTOR_TO_HUMAN_READABLE = {
    'dp': 'Dynamic Programming',
    'dbn': 'Dynamic Bayesian Network',
    'crf': 'Conditional Random Fields',
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