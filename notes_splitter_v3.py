import numpy as np
import librosa
import soundfile as sf

def process_audio(file_path, sample_rate=44100, config_name='piano'):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Instrument configurations
    configs = {
        'piano': {
            'note_ranges': [
                {'range': range(0, 19), 'duration': 4, 'silence': 2},  # A0 to F#2
                {'range': range(19, 52), 'duration': 3, 'silence': 1},  # G2 to E5
                {'range': range(52, 70), 'duration': 2, 'silence': 2},  # F5 to F#6
                {'range': range(70, 88), 'duration': 1, 'silence': 3}  # G6 to C8
            ],
            'notes_count': 88
        },
        'guitar': {
            'note_ranges': [
                {'range': range(0, 33), 'duration': 3, 'silence': 1},
                {'range': range(33, 44), 'duration': 2, 'silence': 2}
            ],
            'notes_count': 44
        },
        'viola': {
            'note_ranges': [
                {'range': range(0, 25), 'duration': 3, 'silence': 1},
                {'range': range(25, 46), 'duration': 2, 'silence': 2}
            ],
            'notes_count': 46
        }
    }

    # Get configuration for the specified instrument
    config = configs[config_name]
    note_ranges = config['note_ranges']
    notes_count = config['notes_count']

    # List to store extracted notes
    extracted_notes = []

    # Process notes
    current_pos = 0
    for note_idx in range(notes_count):
        print(f'Processing note {note_idx + 1} / {notes_count}')
        # Determine the current note range
        current_range = next(r for r in note_ranges if note_idx in r['range'])
        duration = current_range['duration']
        silence = current_range['silence']

        # Extract note samples
        note_end = current_pos + duration * sample_rate
        note_samples = y[current_pos:note_end]

        # Store the raw note samples
        extracted_notes.append(note_samples)

        # Move to the next note
        current_pos = note_end + silence * sample_rate

    return extracted_notes, sample_rate

def save_notes(notes, sample_rate, instrument, output_dir):
    for i, note in enumerate(notes):
        filename = f'{output_dir}/{instrument}_note_{i+1}.wav'
        sf.write(filename, note, sample_rate)

# Usage
instrument = 'piano'
file_path = f'audio/{instrument}/remastered_v2/notes_{instrument}_long.wav'
output_dir = f'audio/{instrument}/remastered_v2/extracted_notes'

# Extract notes
extracted_notes, sample_rate = process_audio(file_path, 11025, config_name=instrument)

# Save individual notes
save_notes(extracted_notes, sample_rate, instrument, output_dir)

print(f"Extracted {len(extracted_notes)} notes for {instrument}")
print(f"Notes saved in {output_dir}")

# Optionally, save all notes as a single numpy array
all_notes = np.array(extracted_notes, dtype=object)
np.save(f'{output_dir}/{instrument}_all_notes.npy', all_notes)
print(f"All notes saved as {instrument}_all_notes.npy")