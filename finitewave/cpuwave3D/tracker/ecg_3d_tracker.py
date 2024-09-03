import os
import numpy as np
from numba import njit, prange
from scipy import spatial

from finitewave.core.tracker.tracker import Tracker


@njit(parallel=True)
def measure(mesh, curr, coord):
    n0 = coord.shape[0]
    n1 = curr.shape[0]
    n2 = curr.shape[1]
    n3 = curr.shape[2]

    ecg = np.zeros(n0)
    for i in prange(n0 * n1 * n2 * n3):
        i0 = i // (n1 * n2 * n3)
        i1 = i % (n1 * n2 * n3) // (n2 * n3)
        i2 = (i % (n1 * n2 * n3)) % (n2 * n3) // n3
        i3 = (i % (n1 * n2 * n3)) % (n2 * n3) % n3
        if mesh[i1, i2, i3] != 1:
            continue
        ecg[i0] += curr[i1, i2, i3] / ((coord[i0, 0] - i1)**2 +
                                       (coord[i0, 1] - i2)**2 +
                                       (coord[i0, 2] - i3)**2)
    return ecg


class ECG3DTracker(Tracker):
    def __init__(self, memory_save=False):
        Tracker.__init__(self)
        # self.radius = radius
        self.measure_coords = np.array([[0, 0, 1]])
        self.ecg = np.ndarray
        self.step = 1
        self._index = 0
        self.memory_save = memory_save

    def initialize(self, model):
        self.model = model
        n = self.measure_coords.shape[0]
        m = int(np.ceil(model.t_max / (self.step * model.dt)))
        self.ecg = np.zeros((n, m), dtype=model.npfloat)
        self.tissue_coords = np.argwhere(model.cardiac_tissue.mesh == 1
                                         ).astype(np.int32)

        if self.memory_save:
            self.uni_voltage = self._uni_voltage_memory_save
            return

        self.compute_distance()

    def compute_distance(self):
        self.distance = np.ones((self.measure_coords.shape[0],
                                 self.tissue_coords.shape[0]),
                                dtype=np.float16)

        for i, point in enumerate(self.measure_coords):
            self.distance[i, :] = np.sum((point - self.tissue_coords)**2,
                                         axis=1).astype(np.float32)

    def uni_voltage(self, current):
        return np.sum(current[tuple(self.tissue_coords.T)] / self.distance,
                      axis=1)

    def _uni_voltage_memory_save(self, current):
        return self.measure(current, self.measure_coords)

    def measure(self, current, coords, batch_size=10):
        ecg = []
        split_inds = np.arange(coords.shape[0])[::batch_size][1:]
        coords = np.split(coords, split_inds)
        for coord in coords:
            distance = spatial.distance.cdist(coord, self.tissue_coords)
            ecg.append(np.sum(current[tuple(self.tissue_coords.T)]
                              / distance ** 2, axis=1))
        ecg = np.hstack(ecg)
        return ecg

    def calc_ecg(self):
        current = self.model.u_new - self.model.u
        current[self.model.cardiac_tissue.mesh != 1] = 0
        return self.uni_voltage(current) / self.model.dr

    def track(self):
        if self.model.step % self.step == 0:
            self.ecg[:, self._index] = self.calc_ecg()
            self._index += 1

    def write(self):
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        np.save(self.ecg)
