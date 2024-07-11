import numpy as np
import librosa
import matplotlib.pyplot as plt

def process_audio(file_path, sample_rate=44100):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Define note durations and silences
    note_ranges = [
        {'range': range(0, 19), 'duration': 4, 'silence': 2},  # A0 to F#2
        {'range': range(19, 52), 'duration': 3, 'silence': 1},  # G2 to E5
        {'range': range(52, 70), 'duration': 2, 'silence': 2},  # F5 to F#6
        {'range': range(70, 88), 'duration': 1, 'silence': 3}  # G6 to C8
    ]

    # Define stage durations in samples
    stage_ranges = {
        'attack': (2112, 2205),
        'decay': (4410, 8820),
        'release': (2205, 4410)
    }

    # Initialize the result matrix
    result = np.zeros((4, 88, 545, 3))

    def calculate_stft(samples):
        stft = librosa.stft(samples, n_fft=1088, hop_length=512)
        mag_stft = np.abs(stft)
        return librosa.util.normalize(mag_stft)

    # Process notes
    current_pos = 0
    for note_idx in range(88):
        print(f"Processing note {note_idx + 1}/{88}")
        # Determine the current note range
        current_range = next(r for r in note_ranges if note_idx in r['range'])
        duration = current_range['duration']
        silence = current_range['silence']

        # Extract note samples
        note_end = current_pos + duration * sample_rate
        note_samples = y[current_pos:note_end]

        # plt.figure(figsize=(16, 8))
        # plt.plot(y)
        # plt.axvline(current_pos, color='r')
        # plt.axvline(note_end, color='r')
        # plt.show()

        # Create three variations
        for var in range(3):
            if var == 0:  # Lower range
                attack = stage_ranges['attack'][0]
                decay = stage_ranges['decay'][0]
                release = stage_ranges['release'][0]
            elif var == 1:  # Middle range
                attack = (stage_ranges['attack'][0] + stage_ranges['attack'][1]) // 2
                decay = (stage_ranges['decay'][0] + stage_ranges['decay'][1]) // 2
                release = (stage_ranges['release'][0] + stage_ranges['release'][1]) // 2
            else:  # Higher range
                attack = stage_ranges['attack'][1]
                decay = stage_ranges['decay'][1]
                release = stage_ranges['release'][1]

            # Calculate sustain duration
            sustain = len(note_samples) - attack - decay

            # Extract and process stages
            attack_samples = note_samples[:attack]
            decay_samples = note_samples[attack:attack + decay]
            sustain_samples = note_samples[attack + decay:attack + decay + sustain]
            release_samples = y[note_end:note_end + release]  # release is after the note_samples part

            # Calculate STFT for each stage
            attack_stft = calculate_stft(attack_samples)
            decay_stft = calculate_stft(decay_samples)
            sustain_stft = calculate_stft(sustain_samples)
            release_stft = calculate_stft(release_samples)

            # plt.figure(figsize=(16, 8))
            # plt.subplot(4,1,1)
            # plt.plot(attack_stft)
            # plt.subplot(4,1,2)
            # plt.plot(decay_stft)
            # plt.subplot(4,1,3)
            # plt.plot(sustain_stft)
            # plt.subplot(4,1,4)
            # plt.plot(release_stft)
            # plt.show()

            # Store results
            result[0, note_idx, :, var] = np.mean(attack_stft, axis=1)[:545]
            result[1, note_idx, :, var] = np.mean(decay_stft, axis=1)[:545]
            result[2, note_idx, :, var] = np.mean(sustain_stft, axis=1)[:545]
            result[3, note_idx, :, var] = np.mean(release_stft, axis=1)[:545]

            print(attack_stft.shape, attack_stft)

        # Move to the next note
        current_pos = note_end + silence * sample_rate

    return result


# Usage
file_path = 'audio/piano/remastered_v2/notes_piano_long.wav'
result_matrix = process_audio(file_path)

# Save the result
np.save('audio/piano/remastered_v2/notes/piano_template.npy', result_matrix)
print("Matrix shape:", result_matrix.shape)
print("Matrix saved as piano_template.npy")