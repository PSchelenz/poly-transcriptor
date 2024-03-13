import madmom
import matplotlib.pyplot as plt
import librosa
import numpy as np


class Madmom:
    def __init__(self, filename: str, sr: int = 44100, hop_length: int = 512):
        self.filename = filename
        self.sr = sr
        self.hop_length = hop_length
        self.onsets = None

    def detect_onsets(self):
        proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=86)
        act = madmom.features.onsets.RNNOnsetProcessor()(self.filename)
        self.onsets = proc(act)

    def plot(self, show=True, color='goldenrod', style='--'):
        # Convert onset times to frame indices
        onset_frames = librosa.time_to_frames(self.onsets, sr=self.sr, hop_length=self.hop_length)

        for onset_frame in onset_frames:
            plt.axvline(x=onset_frame, color=color, linestyle=style, linewidth=1,
                        label='Onset - RNN' if 'Onset - RNN' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.legend()

        if show:
            plt.show()
