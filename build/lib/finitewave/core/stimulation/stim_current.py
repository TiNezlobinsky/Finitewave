from finitewave.core.stimulation.stim import Stim


class StimCurrent:
    def __init__(self, time, curr_value, curr_time):
        Stim.__init__(self, time)
        self.curr_value = curr_value
        self.curr_time = curr_time

        self._acc_time = curr_time
        self._dt = 0

    def ready(self, model):
        self._acc_time = self.curr_time
        self._dt = model.dt
        self.passed = False

    def done(self):
        if self._acc_time >= 0:
            self._acc_time -= self._dt
        else:
            self.passed = True
