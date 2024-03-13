import os
from natsort import natsorted

# Path to the folder containing the files to be renamed
folder_path = '../samples/piano/renamed'  # Change this to the path of your folder

# Tuple of tuples with file renaming mapping (current_name, new_name)
notes = ['C', 'Cis', 'D', 'Dis', 'E', 'F', 'Fis', 'G', 'Gis', 'A', 'Ais', 'H']
octaves = ['2', '3', '4', '5', '6', '7', '8']
phases = ['attack', 'decay', 'sustain']

def file_generator():
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for current_file in natsorted(files):
        yield os.path.join(folder_path, current_file)

generator = file_generator()
doBreak = False

for octave in octaves:
    for note in notes:
        if octave == '2' and (note == 'C' or note == 'Cis'):
            continue
        for phase in phases:
            try:
                current_file_path = next(generator)
                new_name = f'{note}{octave}_{phase}.wav'
                new_file_path = os.path.join(folder_path, new_name)
                os.rename(current_file_path, new_file_path)
            except StopIteration:
                doBreak = True
                break

        if doBreak:
            break

    if doBreak:
        break

