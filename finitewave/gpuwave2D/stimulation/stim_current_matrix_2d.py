import numpy as np

from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentMatrix2D(StimCurrent):
    def __init__(self, time, curr_value, curr_time, matrix):
        StimCurrent.__init__(self, time, curr_value, curr_time)
        self.matrix = matrix.astype('float32')

    def stimulate(self, model):
        if not self.passed:
            value = np.float32(self._dt * self.curr_value)
            model.stim_curr(self.matrix, value)
