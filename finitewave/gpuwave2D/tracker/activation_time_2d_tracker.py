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
        self.model.threshold = np.float32(self.threshold)

    def track(self):
        if self.model.t >= self.model.t_max - 1.5*self.model.dt:
            self.model.track_act_time()
            self.model.get_array('act_t')
            self.act_t = self.model.act_t
        else:
            self.model.track_act_time()

    @property
    def output(self):
        return self.act_t

    def write(self):
        np.save(os.path.join(self.path, self.file_name), self.act_t)
