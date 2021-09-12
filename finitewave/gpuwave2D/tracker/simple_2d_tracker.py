import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class Simple2DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.target_array = ""

    def initialize(self, model):
        self.model = model

    def track(self):
        if self.model.t >= self.model.t_max - 1.5*self.model.dt:
            self.model.get_array(self.target_array)
