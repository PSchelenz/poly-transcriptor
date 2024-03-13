# A new method for Detecting Onset and Offset for Singing in Real-Time and Offline Environments
# https://www.mdpi.com/2076-3417/12/15/7391

import numpy as np
import matplotlib.pyplot as plt

STATUSES = {
    'None': 0,
    'Onset': 1,
    'Start transition': 2,
    'Transition': 3,
    'End transition': 4,
    'Offset': 5
}


class NewMethod:
    def __init__(self, f0: np.ndarray, sr: int, frame_length: int):
        self.mean_window_size = 5
        self.std_tolerance = 2  # minimum is 1
        self.frame_length = frame_length
        self.sr = sr
        self.f0 = f0
        self.frame_in_ms = (frame_length / sr) * 1000
        self.stretched_f0 = None
        self.sloped_f0 = None
        self.status_f0 = None

    def stretch_pitch(self, f0: np.ndarray):
        stretched_f0 = np.empty(np.size(f0))
        f_max = max(f0)
        # threshold = np.ceil(f_max)
        threshold = 1000

        for i, f in enumerate(f0):
            stretched_f0[i] = (f * threshold) / f_max

        return stretched_f0

    def detect_slopes(self, stretched_f0: np.ndarray, frame_in_ms: float):
        sloped_f0 = np.zeros(
            (np.size(stretched_f0), 4))  # hz, own_slope, following_slope, number_of_same_slope_direction
        sloped_f0[0] = [stretched_f0[0], 0, 0, 1]  # starting point has no slope
        f_before = stretched_f0[0]

        for i, f_current in enumerate(
                stretched_f0[1:-1]):  # last element is handled separately at the end of the function
            slope_sum = self.slope(f_before, f_current, frame_in_ms)
            current_slope = self.slope(f_before, f_current, frame_in_ms)

            num_of_same_slope_direction = 1

            f_current_copy = f_current

            for j, f_next in enumerate(stretched_f0[i + 2:]):
                next_slope = self.slope(f_current_copy, f_next, frame_in_ms)

                if (current_slope > 0 and next_slope > 0) or (current_slope < 0 and next_slope < 0):
                    slope_sum += next_slope
                    num_of_same_slope_direction += 1

                    f_current_copy = f_next
                else:
                    sloped_f0[i + 1] = [stretched_f0[i + 1], current_slope, slope_sum, num_of_same_slope_direction]
                    break

            f_before = f_current

        last_slope = self.slope(stretched_f0[-2], stretched_f0[-1], frame_in_ms)
        sloped_f0[-1] = [stretched_f0[-1], last_slope, last_slope, 1]  # ending point has only its own slope

        return sloped_f0

    def slope(self, f_before: float, f_after: float, frame_in_ms: float):
        return (f_after - f_before) / frame_in_ms

    # TODO: somehow this function counts sum of n+1 elements, but the paper divides this sum by 'n' and calls it 'mean'
    def slope_mean(self, point_index: int, sloped_f0: np.ndarray, n: int):
        starting_index = point_index - n

        slopes_sum = 0
        for i in range(starting_index,
                       point_index + 1):  # point index is included but range doesn't include the last element, so we add 1 to it
            slopes_sum += sloped_f0[i][1]  # add own slope to the sum

        return slopes_sum / n

    def slope_std(self, point_index: int, sloped_f0: np.ndarray, mean_f0: np.ndarray, n: int):
        starting_index = point_index - n

        slopes_sum = 0
        for i in range(starting_index,
                       point_index + 1):  # point index is included but range doesn't include the last element, so we add 1 to it
            slopes_sum += (sloped_f0[i][1] - mean_f0[
                point_index - n]) ** 2  # add own slope to the sum; mean_f0 is smaller than sloped_f0 by n which is the "self.mean_window_size"

        return np.sqrt(slopes_sum / (n - 1))

    def detect_onset_offset(self):
        self.stretched_f0 = self.stretch_pitch(self.f0)
        self.sloped_f0 = self.detect_slopes(self.stretched_f0, self.frame_in_ms)

        self.status_f0 = np.zeros(np.size(self.sloped_f0, axis=0))
        mean_f0 = np.zeros(
            np.size(self.sloped_f0,
                    axis=0))  # not necessary to be a separate array, but it allows counting mean only once

        first_time = False
        jump_iterations = 0

        for i in range(self.mean_window_size, np.size(self.sloped_f0, axis=0)):
            if jump_iterations:
                jump_iterations -= 1
                continue

            if not first_time and self.sloped_f0[i][0] == 0:
                first_time = True

            mean_f0[i] = self.slope_mean(i, self.sloped_f0, self.mean_window_size)
            std = self.slope_std(i, self.sloped_f0, mean_f0, self.mean_window_size)

            if abs(self.sloped_f0[i][2]) > mean_f0[i] + (self.std_tolerance * std):  # threshold
                if first_time:
                    self.status_f0[i] = 2  # start transition

                    j = int(self.sloped_f0[i][3])  # number_of_same_slope_direction

                    self.status_f0[i:i + j - 1] = 3  # transition
                    self.status_f0[i + j - 1] = 4  # end transition
                    self.status_f0[i + j] = 1  # onset

                    jump_iterations = j

                    first_time = False
                else:
                    self.status_f0[i + 1] = 2  # start transition

                    j = int(self.sloped_f0[i][3])  # number_of_same_slope_direction

                    self.status_f0[i + 1:i + j - 1] = 3  # transition
                    self.status_f0[i + j - 1] = 4  # end transition
                    self.status_f0[i] = 5  # offset, TODO: zmieniłem kolejność, bo algorytm nie przewidział, że onset może być w następnym punkcie po offset, przez co był nadpisywany przez transition
                    self.status_f0[i + j] = 1  # onset

                    jump_iterations = j
            else:
                self.status_f0[i] = 0

    def plot(self, show: bool = True, xx: np.ndarray = None, colors: list = ('limegreen', 'indianred'), styles: list = ('-', '-')):
        for x, status in zip(xx, self.status_f0):
            if status == 1:
                plt.axvline(x, color=colors[0], linestyle=styles[0], linewidth=1, label='Onset - new method' if 'Onset - new method' not in plt.gca().get_legend_handles_labels()[1] else '')
            elif status == 5:
                plt.axvline(x, color=colors[1], linestyle=styles[1], linewidth=1, label='Offset - new method' if 'Offset - new method' not in plt.gca().get_legend_handles_labels()[1] else '')

        if show:
            plt.show()
