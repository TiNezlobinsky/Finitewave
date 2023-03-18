from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentMatrix2D(StimCurrent):
    def __init__(self, time, curr_value, curr_time, matrix):
        StimCurrent.__init__(self, time, curr_value, curr_time)
        self.matrix = matrix

    def stimulate(self, model):
        if not self.passed:
            model.u[self.matrix ] += self._dt*self.curr_value
