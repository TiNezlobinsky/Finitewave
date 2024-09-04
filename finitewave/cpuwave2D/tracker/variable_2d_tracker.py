import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class Variable2DTracker(Tracker):
    """
    A tracker that records the values of specified variables from a 2D model 
    over time at a given grid point.

    Parameters
    ----------
    var_list : list of str
        List of variable names to be tracked.
    cell_ind : list of int
        Indices of the cell to track. Default is [1, 1].
    dir_name : str
        Directory name where the data will be saved. Default is "multi_vars".
    vars : dict
        Dictionary to store the tracked variables over time.

    Attributes
    ----------
    model : object
        The model object from which data is being tracked.
    """
    def __init__(self):
        Tracker.__init__(self)
        self.var_list = []
        self.cell_ind = [1, 1]
        self.dir_name = "multi_vars"
        self.vars = {}

    def initialize(self, model):
        """
        Initializes the tracker with the given model.

        Parameters
        ----------
        model : object
            The model object from which data is being tracked. It must have attributes
            `t_max` (total simulation time) and `dt` (time step) to determine the length
            of the tracking arrays.
        """
        self.model = model
        t_max = self.model.t_max
        dt    = self.model.dt
        for var_ in self.var_list:
            self.vars[var_] = np.zeros(int(t_max/dt)+1)

    def track(self):
        """
        Updates the tracked variable values at the specified cell index for the current step.
        """
        step  = self.model.step
        for var_ in self.var_list:
            self.vars[var_][step] = self.model.__dict__[var_][self.cell_ind[0],
                                                              self.cell_ind[1]]

    def write(self):
        """
        Saves the tracked variable data to files in the specified directory.
        Creates the directory if it does not exist.
        """
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        for var_ in self.var_list:
            np.save(os.path.join(self.dir_name, var_), self.vars[var_])
