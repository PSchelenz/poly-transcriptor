import librosa
import soundfile as sf
import numpy as np
import os


def split_wav_into_notes(input_file, output_dir):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)

    # Calculate the number of samples per second
    samples_per_second = sr

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the total number of notes (assuming 1 second per note)
    total_notes = len(y) // samples_per_second

    # Define the note names in the correct piano sequence
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    # Generate all note names from A0 to C8 in the correct piano sequence
    all_notes = []
    for octave in range(0, 8):
        if octave == 0:
            all_notes.extend([f"{note}0" for note in note_names[:3]])
        else:
            all_notes.extend([f"{note}{octave}" for note in note_names])
    all_notes.append('C8')  # Add the final C8

    # Split the audio into individual notes and save them
    for i in range(total_notes):
        start_sample = i * samples_per_second
        end_sample = (i + 1) * samples_per_second
        note_audio = y[start_sample:end_sample]

        # Get the corresponding note name
        if i < len(all_notes):
            note_name = all_notes[i]
        else:
            note_name = f"unknown_{i}"

        # Save the note as a separate WAV file
        output_file = os.path.join(output_dir, f"{note_name}.wav")
        sf.write(output_file, note_audio, sr)

        print(f"Saved {output_file}")


# Usage
input_file = "audio/piano/remastered/notes-together.wav"
output_dir = "audio/piano/remastered/notes"
split_wav_into_notes(input_file, output_dir)