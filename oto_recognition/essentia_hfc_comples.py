import essentia
from essentia.standard import *
import matplotlib.pyplot as plt


class Essentia:
    def __init__(self, filename: str, frame_rate: int = 44100, hop_size: int = 512, frame_size: int = 2048):
        self.filename = filename
        self.frame_rate = frame_rate
        self.hop_size = hop_size
        self.frame_size = frame_size
        self.audio = None
        self.onsets_hfc_indices = []
        self.onsets_complex_indices = []
        self.pool = essentia.Pool()
        self.hfc_enabled = False
        self.complex_enabled = False

    def use_hfc(self):
        self.hfc_enabled = True

    def use_complex(self):
        self.complex_enabled = True

    def load_audio(self):
        self.audio = MonoLoader(filename=self.filename, sampleRate=self.frame_rate)()

    def compute_onset_functions(self):
        if self.hfc_enabled:
            od_hfc = OnsetDetection(method='hfc')

        if self.complex_enabled:
            od_complex = OnsetDetection(method='complex')

        if not self.hfc_enabled and not self.complex_enabled:
            self.use_hfc()
            od_hfc = OnsetDetection(method='hfc')

        w = Windowing(type='hann')
        fft = FFT()
        c2p = CartesianToPolar()

        frames = self.generate_frames()

        for frame in frames:
            mag, phase = c2p(fft(w(frame)))

            if self.hfc_enabled:
                self.pool.add('features.hfc', od_hfc(mag, phase))

            if self.complex_enabled:
                self.pool.add('features.complex', od_complex(mag, phase))

    def generate_frames(self):
        return FrameGenerator(self.audio, frameSize=self.frame_size, hopSize=self.hop_size)

    def detect_onsets(self):
        self.load_audio()
        self.compute_onset_functions()

        onsets = Onsets()

        frame_rate = self.frame_rate / self.hop_size

        if self.hfc_enabled:
            onsets_hfc = onsets(essentia.array([self.pool['features.hfc']]), [1])
            self.onsets_hfc_indices = [int(round(onset_time * frame_rate)) for onset_time in onsets_hfc]

        if self.complex_enabled:
            onsets_complex = onsets(essentia.array([self.pool['features.complex']]), [1])
            self.onsets_complex_indices = [int(round(onset_time * frame_rate)) for onset_time in onsets_complex]

    def plot(self, show=True, colors: list = ('darkorchid', 'indigo'), styles: list = ('-.', '-.')):
        if self.hfc_enabled:
            for idx in self.onsets_hfc_indices:
                plt.axvline(x=idx, color='red', linestyle='-.', linewidth=1,
                            label='Onset - HFC' if idx == self.onsets_hfc_indices[0] else "")

        if self.complex_enabled:
            for idx in self.onsets_complex_indices:
                plt.axvline(x=idx, color='green', linestyle='-.', linewidth=1,
                            label='Onset - Complex' if idx == self.onsets_complex_indices[0] else "")

        if show:
            plt.show()
