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
        self.model     = model
        self.act_t     = [-np.ones(self.model.u.shape)]
        self.activated = np.full(self.model.u.shape, True)
        self.activated[1:-1, 1:-1] = False
        self.amount    = np.ones(self.model.u.shape)

    def track(self):

        updated_array = np.where((self.act_t[-1] < 0) & (self.model.u > self.threshold), self.model.t, -1)

        if np.any((self.activated == False) & (self.act_t[-1] > 0) & (self.model.u > self.threshold)):
            self.amount = np.where((self.activated == False) & (self.act_t[-1] > 0) & (self.model.u > self.threshold),
                                    self.amount + 1, self.amount)
            if np.any(self.amount > len(self.act_t)):
                self.act_t.append(updated_array)
        else:
            self.act_t[-1] = np.where(updated_array > 0, updated_array, self.act_t[-1])

        self.activated[1:-1, 1:-1] = np.where((self.model.u[1:-1, 1:-1] > self.threshold) & (self.activated[1:-1, 1:-1] == False), True, self.activated[1:-1, 1:-1])
        self.activated[1:-1, 1:-1] = np.where((self.model.u[1:-1, 1:-1] <= self.threshold) & (self.activated[1:-1, 1:-1] == True), False, self.activated[1:-1, 1:-1])


    @property
    def output(self):
        return self.act_t

    def write(self):
        np.save(os.path.join(self.path, self.file_name), self.act_t)
