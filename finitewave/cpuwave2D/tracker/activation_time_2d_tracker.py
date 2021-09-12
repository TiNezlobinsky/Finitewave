import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class ActivationTime2DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.act_t = np.array([])
        self.threshold = -40
        self.file_name = "act_time_2d"

    def initialize(self, model):
        self.model = model
        self.act_t = -np.ones(self.model.u.shape)

    def track(self):
        self.act_t = np.where(np.logical_and(self.act_t < 0,
                                             self.model.u > self.threshold),
                              self.model.t,
                              self.act_t)

    @property
    def output(self):
        return self.act_t

    def write(self):
        np.save(os.path.join(self.path, self.file_name), self.act_t)
