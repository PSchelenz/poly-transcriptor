import librosa
import librosa.display
import madmom
import numpy as np
import matplotlib.pyplot as plt

'''
Paul Masri, “Computer Modeling of Sound for Transformation and Synthesis of Musical Signals”, PhD thesis, University of Bristol, 1996.
'''

def detect_onsets_sf(filename, sr=44100, hop_length=512, frame_length=2048, max_draw_note=88):
    wlazl_kotek = {
        "pre_avg": 0.1,
        "post_avg": 0.1,
        "pre_max": 0.03,
        "post_max": 0.03,
        "combine": 0.03,
        "threshold": 8
    }

    # Dictionary for "Lot trzmiela"
    lot_trzmiela = {
        "pre_avg": 0.08,
        "post_avg": 0.08,
        "pre_max": 0.08,
        "post_max": 0.08,
        "combine": 0.07,
        "threshold": 0.1
    }

    # Dictionary for "Kołysanka"
    kolysanka = {
        "pre_avg": 1,
        "post_avg": 1,
        "pre_max": 0.5,
        "post_max": 0.5,
        "combine": 0.5,
        "threshold": 1
    }

    # Dictionary for "Kołysanka 2" (identical to "Kołysanka" based on provided information)
    kolysanka_2 = {
        "pre_avg": 1,
        "post_avg": 1,
        "pre_max": 0.5,
        "post_max": 0.5,
        "combine": 0.5,
        "threshold": 12
    }

    kolysanka_3 = {
        "pre_avg": 1,
        "post_avg": 1,
        "pre_max": 0.5,
        "post_max": 0.5,
        "combine": 0.5,
        "threshold": 3
    }

    dictionary = kolysanka_2

    proc = madmom.features.onsets.OnsetPeakPickingProcessor(
        pre_avg=dictionary["pre_avg"],
        post_avg=dictionary["post_avg"],
        pre_max=dictionary["pre_max"],
        post_max=dictionary["post_max"],
        combine=dictionary["combine"],
        threshold=dictionary["threshold"],
        fps=100
    )
    act = madmom.features.onsets.SpectralOnsetProcessor('spectral_flux',
                                                        sample_rate=sr,
                                                        frame_size=frame_length,
                                                        hop_size=hop_length)(filename)

    # Detect onsets
    onsets = proc(act)
    onsets = onsets * 100

    # Time vector for the audio segment
    # time = np.linspace(0, 30, num=len(audio))

    # Plot the audio waveform of the segment and detected onsets
    # plt.figure(figsize=(14,4))
    # plt.plot(time, audio, label='Audio Waveform (Segment)')
    # plt.vlines(onsets, ymin=0, ymax=max_draw_note, color='r', linestyle='--', label='Detected Onsets')
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Detected Onsets in Polyphonic Music (30-second Segment)')
    # plt.show()

    return onsets

if __name__ == '__main__':
    # Replace 'path/to/your/music/file' with the actual path to your audio file
    onset_times = detect_onsets_sf('../../audio/midi_tracks/Canon_in_D.mp3', 44100, 512)
    plt.show()