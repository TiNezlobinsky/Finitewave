import os
import numpy as np

from finitewave.core.tracker import Tracker


class PeriodMesh2DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.period_count = np.ndarray
        self.time_stop = np.ndarray
        self.time_start = np.ndarray
        self.state = np.ndarray
        self.threshold = 0.5
        self.file_name = "period_mesh.npy"

    def initialize(self, model):
        self.model = model
        self.time_start = np.zeros_like(self.model.u)
        self.time_stop = np.zeros_like(self.model.u)
        self.state = np.ones_like(self.model.u, dtype=bool)
        self.period_count = np.zeros_like(self.model.u, dtype=int)

    def track(self):
        mask = (self.model.u > self.threshold) & (self.state == 1)
        self.time_start[mask & (self.period_count == 0)] = self.model.t
        self.time_stop[mask] = self.model.t
        self.period_count[mask] += 1
        self.state[mask] = 0
        self.state[self.model.u < self.threshold] = 1

    @property
    def output(self):
        out = np.zeros_like(self.time_start)
        np.divide(self.time_stop - self.time_start, self.period_count,
                  where=self.period_count > 0, out=out)
        return out
