import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class Variable3DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.var_list = []
        self.cell_ind = [1, 1, 1]
        self.dir_name = "multi_vars"
        self.vars = {}

    def initialize(self, model):
        self.model = model
        t_max = self.model.t_max
        dt    = self.model.dt
        for var_ in self.var_list:
            self.vars[var_] = np.zeros(int(t_max/dt)+1)

    def track(self):
        step  = self.model.step
        for var_ in self.var_list:
            self.vars[var_][step] = self.model.__dict__[var_][self.cell_ind[0],
                                                              self.cell_ind[1],
                                                              self.cell_ind[2]]

    def write(self):
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        for var_ in self.var_list:
            np.save(os.path.join(self.dir_name, var_), self.vars[var_])
