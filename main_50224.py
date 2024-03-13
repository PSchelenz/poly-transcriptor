import librosa
import matplotlib.pyplot as plt
import numpy as np
from pitch_recognition.pyin import PYin
from oto_recognition.new_method import NewMethod
from oto_recognition.madmom_onset import Madmom
from oto_recognition.essentia_hfc_comples import Essentia

if __name__ == "__main__":
    filename = librosa.example('trumpet')
    sampling_rate = 44100
    hop_size = 512
    frame_size = 2048

    plt.figure(figsize=(20, 6))
    plt.xlabel('Frame Index')
    plt.ylabel('Frequency (Hz)')
    plt.title('Fundamental Frequency with Onsets (and Offsets)')

    '''PYIN + NEW METHOD'''
    pyin = PYin(filename, sampling_rate, frame_size, hop_size)
    f0, _ = pyin.get_pitch_and_sr()[:2]
    xx = pyin.plot(False)

    new_method = NewMethod(f0, sampling_rate, frame_size)
    new_method.detect_onset_offset()
    new_method.plot(False, xx)

    '''PYIN + MADMOM (RNN)'''
    pyin = PYin(filename, sampling_rate, frame_size, hop_size)
    f0, _ = pyin.get_pitch_and_sr()[:2]

    mmom = Madmom(filename, sampling_rate, hop_size)
    mmom.detect_onsets()
    mmom.plot(False)

    '''PYIN + Essentia (HFC and Complex Domain Spectral Difference)'''
    pyin = PYin(filename, sampling_rate, frame_size, hop_size)
    f0, _ = pyin.get_pitch_and_sr()[:2]

    essentiaOnsetDetector = Essentia(filename, sampling_rate, hop_size, frame_size)
    essentiaOnsetDetector.use_hfc()
    essentiaOnsetDetector.use_complex()
    essentiaOnsetDetector.detect_onsets()
    essentiaOnsetDetector.plot(False)

    plt.legend()
    plt.show()
