import numpy as np

from finitewave.core.stimulation import Stim


class StimCurrentCoord2D(Stim):
    def __init__(self, time, current, duration, x1, x2, y1, y2):
        Stim.__init__(self, time, current=current, duration=duration)
        x = np.arange(x1, x2)
        y = np.arange(y1, y2)
        xx, yy = np.meshgrid(x, y)

        self.coords = np.array([xx.ravel(), yy.ravel()]).T

    def stimulate(self, model):
        model.u[self.coords] += model.dt * self.current
