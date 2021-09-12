from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentCoord2D(StimCurrent):
    def __init__(self, time, curr_value, curr_time, x1, x2, y1, y2):
        StimCurrent.__init__(self, time, curr_value, curr_time)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def stimulate(self, model):
        if not self.passed:
            model.u[self.x1:self.x2, self.y1:self.y2] += self._dt*self.curr_value
