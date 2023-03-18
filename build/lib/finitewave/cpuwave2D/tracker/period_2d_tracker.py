import numpy as np
from numba import njit
import json

from finitewave.core.tracker.tracker import Tracker


@njit
def _track_detectors_period(periods, detectors, detectors_state, u, t, threshold, step):
    n_i, n_j = u.shape
    for i in range(n_i):
        for j in range(n_j):
            if detectors[i, j] and u[i, j] > threshold and detectors_state[i, j]:
                periods[step] = [i, j, t]
                detectors_state[i, j] = 0
                step += 1
            elif detectors[i, j] and u[i, j] <= threshold and not detectors_state[i, j]:
                detectors_state[i, j] = 1

    return periods, detectors_state, step


class Period2DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)

        self.detectors = np.array([])
        self.threshold = -40

        self._periods         = np.array([])
        self._detectors_state = np.array([])
        self._step            = 0

        self.file_name = "period"

    def initialize(self, model):
        self.model = model

        t_max = model.t_max
        dt    = model.dt
        n     = 20*len(self.detectors[self.detectors == 1]) # a start length of the array
        self._periods         = -1*np.ones([n, 3])
        self._detectors_state = np.ones(model.u.shape, dtype="uint8")

    def track(self):
        # dynamically increase the size of the array if there is no free space:
        if self._step == len(self._periods):
            self._periods = np.tile(self._periods, (2, 1))
            self._periods[len(self._periods)//2:, :] = -1.

        self.period_detectors, self._detectors_state, self._step = _track_detectors_period(
            self._periods, self.detectors, self._detectors_state,
            self.model.u, self.model.t, self.threshold,
            self._step)

    def compute_periods(self):
        periods_dict = dict()
        to_str = lambda i, j: str(int(i)) + "," + str(int(j))

        for i in range(len(self._periods)):
            if self._periods[i][0] < 0:
                continue
            key = to_str(*self._periods[i][:2])
            if not key in periods_dict:
                periods_dict[key] = []
            periods_dict[key].append(self._periods[i][2])

        for key in periods_dict:
            time_per_list = []
            for i, t in enumerate(periods_dict[key]):
                if i == 0:
                    time_per_list.append([t, 0])
                else:
                    time_per_list.append([t, t-time_per_list[i-1][0]])
            periods_dict[key] = time_per_list

        return periods_dict

    @property
    def output(self):
        return self.compute_periods()

    def write(self):
        jdata = json.dumps(self.compute_periods())
        with open(os.path.join(self.path, self.file_name), "w") as jf:
            jf.write(jdata)
