import numpy as np

from finitewave.core.stimulation import Stim


class StimVoltages3D(Stim):
    def __init__(self, time, duration, voltage, coords):
        Stim.__init__(self, time, duration=duration)
        self._step = 0
        self._voltage = voltage
        self.coords = coords

    def stimulate(self, model):
        model.u[self.coords] = self._voltage[self._step]
        self._step += 1
