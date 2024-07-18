import numpy as np
import librosa


def process_audio(file_path, sample_rate=44100):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    piano_note_ranges = [
        {'range': range(0, 19), 'duration': 4, 'silence': 2},  # A0 to F#2
        {'range': range(19, 52), 'duration': 3, 'silence': 1},  # G2 to E5
        {'range': range(52, 70), 'duration': 2, 'silence': 2},  # F5 to F#6
        {'range': range(70, 88), 'duration': 1, 'silence': 3}  # G6 to C8
    ]

    guitar_note_ranges = [
        {'range': range(0, 33), 'duration': 3, 'silence': 1},
        {'range': range(33, 44), 'duration': 2, 'silence': 2}
    ]

    viola_note_ranges = [
        {'range': range(0, 25), 'duration': 3, 'silence': 1},
        {'range': range(25, 46), 'duration': 2, 'silence': 2}
    ]

    # Define note durations and silences
    note_ranges = piano_note_ranges

    features_count = 3

    # Define stage durations in samples
    stage_ranges = {
        'attack': np.linspace(2112, 2205, features_count).astype(int),
        'decay': np.linspace(4410, 8820, features_count).astype(int),
        'release': np.linspace(2205, 4410, features_count).astype(int)
    }

    # Initialize the result matrix
    result = np.zeros((4, 88, 545, features_count))

    def calculate_cqt(samples):
        # Calculate CQT
        cqt = np.abs(librosa.cqt(samples, sr=sample_rate, n_bins=545, bins_per_octave=74))
        return cqt

    # Process notes
    current_pos = 0
    for note_idx in range(88):
        # Determine the current note range
        print(note_idx)
        current_range = next(r for r in note_ranges if note_idx in r['range'])
        duration = current_range['duration']
        silence = current_range['silence']

        # Extract note samples
        note_end = current_pos + duration * sample_rate
        note_samples = y[current_pos:note_end]

        # Create three variations
        for var, (attack, decay, release) in enumerate(zip(stage_ranges['attack'], stage_ranges['decay'], stage_ranges['release'])):
            # Calculate sustain duration
            sustain = len(note_samples) - attack - decay - release

            # Ensure sustain is not negative
            if sustain < 0:
                sustain = 0
                release = len(note_samples) - attack - decay

            # Extract and process stages
            attack_samples = note_samples[:attack]
            decay_samples = note_samples[attack:attack + decay]
            sustain_samples = note_samples[attack + decay:attack + decay + sustain]
            release_samples = note_samples[-release:]

            # Calculate CQT for each stage
            attack_cqt = calculate_cqt(attack_samples)
            decay_cqt = calculate_cqt(decay_samples)
            sustain_cqt = calculate_cqt(sustain_samples)
            release_cqt = calculate_cqt(release_samples)

            # Store results
            result[0, note_idx, :, var] = np.mean(attack_cqt, axis=1)
            result[1, note_idx, :, var] = np.mean(decay_cqt, axis=1)
            result[2, note_idx, :, var] = np.mean(sustain_cqt, axis=1)
            result[3, note_idx, :, var] = np.mean(release_cqt, axis=1)

        # Move to the next note
        current_pos = note_end + silence * sample_rate

    axis_sum = np.sum(result, axis=2, keepdims=True)
    weights = result / (axis_sum + 1e-8)  # Adding small epsilon to avoid division by zero

    return weights

file_path = 'audio/piano/remastered_v2/notes_piano_long.wav'
result_matrix = process_audio(file_path)

# Save the result
np.save('audio/piano/remastered_v2/notes/piano_template_3r.npy', result_matrix)
print("Matrix shape:", result_matrix.shape)
print("Matrix saved as piano_template_7r.npy")