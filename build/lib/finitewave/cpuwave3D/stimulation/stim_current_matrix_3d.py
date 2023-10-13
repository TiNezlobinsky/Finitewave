import numpy as np

from finitewave.core.stimulation import Stim


class StimCurrentMatrix3D(Stim):
    def __init__(self, time, current, duration, matrix):
        Stim.__init__(self, time, duration=duration)
        self.coords = np.argwhere(matrix > 0)
        self.current = current[tuple(self.coords.T)]

    def stimulate(self, model):
        model.u[self.coords] += model.dt * self.current
