import numpy as np

from finitewave.core.stimulation import Stim


class StimVoltageMatrix3D(Stim):
    def __init__(self, time, voltage, matrix):
        Stim.__init__(self, time)
        self.coords = np.argwhere(matrix > 0)
        self.voltage = voltage[tuple(self.coords.T)]

    def stimulate(self, model):
        model.u[self.coords] = self.voltage
