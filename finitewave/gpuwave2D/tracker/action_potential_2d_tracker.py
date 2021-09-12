import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class ActionPotential2DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.act_pot  = np.array([])
        self.cell_ind = [1, 1]
        self.file_name = "act_pot"

    def initialize(self, model):
        self.model = model
        t_max = self.model.t_max
        dt    = self.model.dt
        self.act_pot = np.zeros(int(t_max/dt)+1)

    def track(self):
        step  = self.model.step
        self.model.get_array("u")
        self.act_pot[step] = self.model.u[self.cell_ind[0],
                                          self.cell_ind[1]]

    @property
    def output(self):
        return self.act_pot

    def write(self):
        np.save(os.path.join(self.path, self.file_name), self.act_pot)
