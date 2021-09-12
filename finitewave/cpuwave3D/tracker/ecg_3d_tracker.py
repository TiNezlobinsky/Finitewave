import os
import numpy as np
from numba import njit, prange
from scipy.spatial import distance

from finitewave.core.tracker.tracker import Tracker


class ECG3DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.measure_points = np.array([[0, 0, 1]])
        self.ecg = np.ndarray
        self.step = 1
        self._index = 0

    def initialize(self, model):
        self.model = model
        n = int(np.ceil(model.t_max / (self.step * model.dt)))
        self.ecg = np.zeros((self.measure_points.shape[0], n))

        mesh = model.cardiac_tissue.mesh
        self.tissue_points = np.where(mesh == 1)

        tissue_points = np.argwhere(mesh == 1)
        self.distances = distance.cdist(self.measure_points, tissue_points)
        self.distances = self.distances**2

    def calc_ecg(self):
        current = (self.model.u_new - self.model.u)[self.tissue_points]
        return np.sum(current / self.distances, axis=1)

    def track(self):
        if self.model.step % self.step == 0:
            self.ecg[:, self._index] = self.calc_ecg()
            self._index += 1

    def write(self):
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        np.save(self.ecg)
