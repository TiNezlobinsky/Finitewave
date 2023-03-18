import numpy as np

from finitewave.core.stimulation import Stim


class StimVoltageReal3D(Stim):
    def __init__(self, time, duration, voltage, x1, x2, y1, y2, z1, z2):
        Stim.__init__(self, time, duration=duration)
        self._step = 0
        self._voltage = voltage

        x = np.arange(x1, x2)
        y = np.arange(y1, y2)
        z = np.arange(z1, z2)
        xx, yy, zz = np.meshgrid(x, y, z)

        self.coords = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T

    def stimulate(self, model):
        model.u[self.coords] = self._voltage[self._step]
        self._step += 1
