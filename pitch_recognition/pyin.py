import librosa
import numpy as np
import matplotlib.pyplot as plt


class PYin:
    def __init__(self, filename: str, sr: int = 44100, frame_length: int = 2048, hop_length: int = 512):
        self.filename = filename
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0 = None

    def get_pitch_and_sr(self):
        y, sr = librosa.load(self.filename, sr=self.sr)

        self.f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                          fmin=librosa.note_to_hz('C2'),
                                                          fmax=librosa.note_to_hz('C7'),
                                                          sr=sr,
                                                          frame_length=self.frame_length,
                                                          hop_length=self.hop_length,
                                                          fill_na=0)

        return self.f0, sr, voiced_flag, voiced_probs

    def plot(self, show=True, color='cornflowerblue', style='-'):
        xx = np.arange(0, len(self.f0))

        plt.plot(xx, self.f0, color=color, linestyle=style, linewidth=2, label='Fundamental Frequency (f0)')

        if show:
            plt.show()

        return xx
