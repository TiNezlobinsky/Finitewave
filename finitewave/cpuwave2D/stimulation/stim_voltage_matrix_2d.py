import numpy as np

from finitewave.core.stimulation import Stim


class StimVoltageMatrix2D(Stim):
    def __init__(self, time, voltage, matrix):
        Stim.__init__(self, time)
        self.coords = np.where(matrix > 0)
        self.voltage = voltage[tuple(self.coords)]

    def stimulate(self, model):
        model.u[self.coords] = self.voltage
