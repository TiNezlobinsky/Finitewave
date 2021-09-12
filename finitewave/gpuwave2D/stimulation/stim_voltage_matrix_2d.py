import numpy as np

from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageMatrix2D(StimVoltage):
    def __init__(self, time, volt_value, matrix):
        StimVoltage.__init__(self, time, volt_value)
        self.matrix = matrix.astype('float32')

    def stimulate(self, model):
        if not self.passed:
            value = np.float32(self.volt_value)
            model.stim_volt(self.matrix, value)
