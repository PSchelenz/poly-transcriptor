import madmom
from madmom.io.audio import load_ffmpeg_file, write_wave_file
import numpy as np
import matplotlib.pyplot as plt

'''
Sebastian Böck and Gerhard Widmer,
"Maximum Filter Vibrato Suppression for Onset Detection",
Proceedings of the 16th International Conference on Digital Audio
Effects (DAFx), 2013

https://www.dafx12.york.ac.uk/papers/dafx12_submission_4.pdf
'''

def detect_onsets_madmom(filename, start_sec, duration_sec):
    # Load only a segment of the audio file
    audio, sr = load_ffmpeg_file(filename, start=start_sec, stop=start_sec + duration_sec)

    # Save the segment to a temporary file because RNNOnsetProcessor needs a file as input
    temp_filename = 'temp_audio_segment.wav'
    write_wave_file(audio, temp_filename, sr)

    # Initialize the pre-trained onset detection model
    proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100)
    act = madmom.features.onsets.RNNOnsetProcessor()(temp_filename)

    # Detect onsets
    onsets = proc(act)

    # Time vector for the audio segment
    time = np.linspace(start_sec, start_sec + duration_sec, num=len(audio))

    # Plot the audio waveform of the segment and detected onsets
    plt.figure(figsize=(14, 6))
    plt.plot(time, audio, label='Audio Waveform (Segment)')
    plt.vlines(onsets, ymin=np.min(audio), ymax=np.max(audio), color='r', linestyle='--', label='Detected Onsets')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Detected Onsets in Polyphonic Music (30-second Segment)')
    plt.show()

    return onsets

# Replace 'path/to/your/music/file' with the actual path to your audio file
# Specify the start time and duration of the segment you want to process (in seconds)
detected_onsets = detect_onsets_madmom('../../audio/midi_tracks/Canon_in_D.mp3', start_sec=0, duration_sec=30)