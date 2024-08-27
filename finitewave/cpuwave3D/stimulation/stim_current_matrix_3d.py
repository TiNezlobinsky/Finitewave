from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentMatrix3D(StimCurrent):
    def __init__(self, time, curr_value, curr_time, matrix):
        StimCurrent.__init__(self, time, curr_value, curr_time)
        self.matrix = matrix

    def stimulate(self, model):
        if not self.passed:
            mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
            model.u[mask] += self._dt*self.curr_value

