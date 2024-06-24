import numpy as np
from mido import MidiFile, tick2second
import string
import matplotlib.pyplot as plt

def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]

def switch_note(last_state, note, velocity, on_=True):
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result

def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

def track2seq(track):
    result = []
    change_times = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
            change_times.append(len(result) - 1)
        last_state, last_time = new_state, new_time
    return result, change_times

def midi2array(filename, min_msg_pct=0.1):
    mid = MidiFile(filename, clip=True)

    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    all_arys = []
    all_change_times = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i, change_times_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
            all_change_times.extend(change_times_i)
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    trimmed_array = all_arys[min(ends): max(ends)]
    adjusted_change_times = [time for time in all_change_times if min(ends) <= time <= max(ends)]
    adjusted_change_times = [time - min(ends) for time in adjusted_change_times]

    unique_change_times = np.unique(adjusted_change_times)
    return trimmed_array, unique_change_times[::2]


if __name__ == "__main__":
    result_array, _ = midi2array('audio/wlazl_kotek/wlazl_kotek.mid')

    plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
    plt.title("Wlazł kotek")
    plt.show()