import madmom
import matplotlib.pyplot as plt
from madmom.audio.filters import LogarithmicFilterbank
from madmom.io.audio import load_ffmpeg_file, write_wave_file
import numpy as np

'''
Sebastian Böck and Gerhard Widmer,
"Maximum Filter Vibrato Suppression for Onset Detection",
Proceedings of the 16th International Conference on Digital Audio
Effects (DAFx), 2013

https://www.dafx12.york.ac.uk/papers/dafx12_submission_4.pdf
'''


def detect_onsets_pd(filename, sr=44100, hop_length=512, frame_length=2048, max_draw_note=88):
    # Load only a segment of the audio file
    # audio, sr = load_ffmpeg_file(filename, start=0, stop=30, sample_rate=44100)

    # Save the segment to a temporary file because RNNOnsetProcessor needs a file as input
    # temp_filename = 'temp_audio_segment.wav'
    # write_wave_file(audio, temp_filename, sr)

    # Initialize the pre-trained onset detection model
    # wlazł kotek - 0.1, 0.1, 0.03, 0.03, 0.03, 0.3
    # lot trzmiela - 0.08, 0.08, 0.08, 0.08, 0.07, 0.1
    # kolysanka - 1.2, 1.2, 0.01, 0.01, 0.5, 0.05
    # kolysanka 2 - 0.5, 0.5, 0.1, 0.1, 0.5, 0.03

    pre_avg = 0.5
    post_avg = 0.5
    pre_max = 0.1
    post_max = 0.1
    combine = 0.5
    threshold = 0.03

    proc = madmom.features.onsets.OnsetPeakPickingProcessor(
        pre_avg=pre_avg,
        post_avg=post_avg,
        pre_max=pre_max,
        post_max=post_max,
        combine=combine,
        threshold=threshold,
        fps=100
    )

    act = madmom.features.onsets.SpectralOnsetProcessor(
        'phase_deviation',
        sample_rate=sr,
        frame_size=frame_length,
        hop_size=hop_length,
        fps=100
    )(filename)

    # Detect onsets
    onsets = proc(act)
    onsets = onsets

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

    return onsets * (sr / hop_length)


if __name__ == '__main__':
    detected_onsets = detect_onsets_pd('../../audio/midi_tracks/Canon_in_D.mp3', 512)
