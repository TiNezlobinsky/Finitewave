import os
import numpy as np

from finitewave.cpuwave3D.tracker import AnimationSlice3DTracker

class PeriodMap3DTracker(AnimationSlice3DTracker):
    def __init__(self):
        AnimationSlice3DTracker.__init__(self)

        self.dir_name = "period"

        self.threshold = -40.
        self.period_map        = np.array([])
        self._period_map_state = np.array([])

    def initialize(self, model):
        AnimationSlice3DTracker.initialize(self, model)

        self.period_map        = -1*np.ones(self.model.u.shape)
        self._last_time_map    = -1*np.ones(self.model.u.shape)
        self._period_map_state = np.ones(self.model.u.shape, dtype="uint8")

    def track(self):
        if self._t > self.step:
            active_nodes = np.logical_and(self._period_map_state == 1, self.model.u > self.threshold)
            self.period_map[active_nodes] = self.model.t - self._last_time_map[active_nodes]
            self._last_time_map[active_nodes] = self.model.t
            self._period_map_state[active_nodes] = 0
            self._period_map_state[np.logical_and(self._period_map_state == 0, self.model.u < self.threshold)] = 1

            np.save(os.path.join(self.path, self.dir_name, str(self._frame_n)), self.period_map)
            self._frame_n += 1
            self._t = 0
        else:
            self._t += self._dt

    def write(self):
        pass
