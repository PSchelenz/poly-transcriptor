import numpy as np
from sklearn.decomposition import NMF
from hmmlearn import hmm
import os
import librosa
from scipy.io import savemat

folder_path = '../samples/piano/renamed'
piano_notes = {}

def get_spectral_template(audio_file: str) -> np.ndarray:
    y, sr = librosa.load(audio_file, sr=44100)

    # nie wszystkie próbki są tej samej długości, np. faza ataku może nie posiadać aż 2048 próbek, żeby móc zastosować
    # STFT z 2048 próbkami, trzeba wtedy uzupełnić próbki zerami. Zanim jednak to to trzeba nałożyć okno Hanninga na
    # prawdziwy sygnał, bo nie chcemy nakładać go na 0 a tym samym zera potrafią wpływać na spectral leakage.
    hann_window = np.hanning(len(y))
    y_windowed = y * hann_window

    # no i uzupełnienie o zera
    if len(y_windowed) < 2048:
        y_padded = np.pad(y_windowed, (0, 2048 - len(y_windowed)), 'constant')
    else:
        y_padded = y_windowed

    #spectrogram
    S = np.abs(librosa.stft(y_padded, n_fft=2048, window='boxcar', center=False))

    # non negative matrix factorization
    model = NMF(n_components=1, init='random', random_state=0)
    W = model.fit_transform(S)
    # H = model.components_ # <- not needed

    return W[:, 0] # ,_ 1 component set in NFM, so we can just return it


def extract_spectral_templates():
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for i, file in enumerate(files):
        spectral_template = get_spectral_template(os.path.join(folder_path, file))
        note, phase_with_extension = file.split('_')
        phase = phase_with_extension.split('.')[0]

        if note not in piano_notes:
            piano_notes[note] = {}

        piano_notes[note][phase] = spectral_template.tolist()

        print(f"Spectral template extraction: {i + 1}/{len(files)}")


#
# def si_plca_decomposition(spectrogram, templates):
#     # Placeholder function for SI-PLCA decomposition
#     return decomposed_components
#
# def apply_hmm_constraints(components):
#     # Placeholder function for applying HMM-based temporal constraints
#     return constrained_components
#
# def estimate_parameters(components):
#     # Placeholder function for parameter estimation using EM algorithm
#     return estimated_parameters
#
# def postprocess_for_note_tracking(estimated_parameters):
#     # Placeholder function for HMM-based note tracking
#     return note_events
#
# # Main algorithm
# def recognize_pitches_polyphonic_music(audio_signal):
#     # Step 1: Spectral template extraction
#     templates = extract_spectral_templates(audio_signal)
#
#     # Step 2: Apply SI-PLCA model
#     spectrogram = compute_spectrogram(audio_signal)
#     components = si_plca_decomposition(spectrogram, templates)
#
#     # Step 3: Apply HMM-based temporal constraints
#     constrained_components = apply_hmm_constraints(components)
#
#     # Step 4: Parameter estimation
#     estimated_parameters = estimate_parameters(constrained_components)
#
#     # Step 5: Postprocessing for note tracking
#     note_events = postprocess_for_note_tracking(estimated_parameters)
#
#     return note_events

def generate_note_sequence(start_octave, end_octave):
    # Define the sequence of notes, considering 'H' as in your notation
    notes = ['C', 'Cis', 'D', 'Dis', 'E', 'F', 'Fis', 'G', 'Gis', 'A', 'Ais', 'H']
    note_sequence = []

    # Generate note names for each octave
    for octave in range(start_octave, end_octave + 1):
        for note in notes:
            if (octave == 2 and (note == 'C' or note == 'Cis')) or (octave == 7 and note == 'H'):
                continue
            note_sequence.append(f"{note}{octave}")

    return note_sequence

note_sequence = generate_note_sequence(2, 7)
note_mapper = {note: i for i, note in enumerate(note_sequence)}

extract_spectral_templates()

notes_to_ints = []
for i in range(len(note_sequence)):
    notes_to_ints.append([])

for note, phases in piano_notes.items():
    notes_to_ints[note_mapper[note]] = [phases['attack'], phases['decay'], phases['sustain']]

dataW = np.array(notes_to_ints)
dataW = dataW[:, :, :, np.newaxis]
dataW = np.transpose(dataW, (1, 0, 2, 3))

mat_dict = {
    'shiftedW': dataW,
}

savemat('pitch_recognition/poly/si_plca/my_shiftedW.mat', mat_dict)