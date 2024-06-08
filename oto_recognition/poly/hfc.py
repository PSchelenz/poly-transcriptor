import numpy as np
import librosa
import matplotlib.pyplot as plt

def detect_onsets_hfc(audio_file, sr=44100, hop_length=512, frame_length=2048, max_draw_note=88):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=sr, duration=30)

    # Preprocessing: Convert to mono and apply pre-emphasis
    y = librosa.to_mono(y)
    y = librosa.effects.preemphasis(y)

    # Framing and FFT
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    # Calculate High Frequency Content (HFC)
    hfc = np.sum(np.arange(S.shape[0])[:, np.newaxis] * S, axis=0)

    # Detect onsets: Compute first-order difference and apply threshold
    onset_env = np.diff(hfc)
    onset_env = np.maximum(0, onset_env)  # Half-wave rectification
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='frames', hop_length=hop_length)
    onset_frames = onset_frames
    # onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    #
    # plt.figure(figsize=(14, 5))
    # librosa.display.waveshow(y, sr=sr, alpha=0.6)
    # plt.vlines(onset_frames, ymin=0, ymax=max_draw_note, color='r', linestyle='--', label='Onsets')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.title('Audio Waveform and Detected Onsets')
    # plt.show()

    # Convert frames to time
    # onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    return onset_frames


if __name__ == '__main__':
    # Example usage
    audio_file = '../../audio/midi_tracks/Canon_in_D.mp3'
    onset_times = detect_onsets_hfc(audio_file)