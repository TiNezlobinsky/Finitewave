import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class MultiActivationTime2DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.act_t = np.array([])
        self.threshold = -40
        self.file_name = "multi_act_time_2d"

    def initialize(self, model):
        self.model = model
        self.act_t = -np.ones((1, *self.model.u.shape))
        self.inactive = ~(self.model.u >= self.threshold)
        self.cross_num = -np.ones_like(self.model.u, dtype=int)

    def track(self):
        threshold_cross = (self.model.u >= self.threshold) & self.inactive
        
        self.inactive[threshold_cross] = False
        self.cross_num[threshold_cross] += 1

        coords = np.argwhere(threshold_cross[np.newaxis, :, :])
        coords[:, 0] = self.cross_num[threshold_cross]

        if self.cross_num.max() >= self.act_t.shape[0]:
            self.act_t = np.vstack((self.act_t, 
                                    -np.ones((1, *self.act_t.shape[1:]))))
        
        self.act_t[tuple(coords.T)] = self.model.t

        back_cross = ~self.inactive & (self.model.u < self.threshold)
        self.inactive[back_cross] = True

    @property
    def output(self):
        return self.act_t

    def write(self):
        np.save(os.path.join(self.path, self.file_name), self.act_t)