import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class Animation2DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.step  = 1
        self.start = 0 
        self._t   = 0

        self.dir_name = "animation"

        self._frame_n = 0

        self.target_array = ""

        self.frame_format = {
          "type" : "float64",
          "mult" : 1
        }

        self._frame_format_type = ""
        self._frame_format_mult = 1

    def initialize(self, model):
        self.model = model

        self._t   = 0
        self._frame_n = 0
        self._dt  = self.model.dt
        self._step = self.step - self._dt

        if not os.path.exists(os.path.join(self.path, self.dir_name)):
            os.makedirs(os.path.join(self.path, self.dir_name))

        self._frame_format_type = self.frame_format["type"]
        self._frame_format_mult = self.frame_format["mult"]       

    def track(self):
        if not self.model.t >= self.start:
            return
        if self._t > self._step:
            frame = (self.model.__dict__[self.target_array]*self._frame_format_mult).astype(self._frame_format_type)
            np.save(os.path.join(self.path, self.dir_name, str(self._frame_n)), frame)
            self._frame_n += 1
            self._t = 0
        else:
            self._t += self._dt


    def write(self):
        pass
