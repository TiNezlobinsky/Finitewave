import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class MultiVariable2DTracker(Tracker):
    """
    A class to track multiple variables at a specific cell in a 2D cardiac tissue model simulation.

    This tracker monitors user-defined variables at a specified cell index and records their values over time.

    Attributes
    ----------
    var_list : list of str
        A list of variable names to be tracked.
    cell_ind : list of int
        The indices [i, j] of the cell where the variables are tracked.
    dir_name : str
        The directory name where tracked variables are saved.
    vars : dict
        A dictionary where each key is a variable name, and the value is an array of its tracked values over time.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model and precomputes necessary values for each variable.
    track():
        Tracks and stores the values of each specified variable at each time step.
    write():
        Saves the tracked variables to disk as NumPy files.
    """

    def __init__(self):
        """
        Initializes the MultiVariable2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.var_list = []  # List of variables to be tracked
        self.cell_ind = [1, 1]  # Cell index to track variables
        self.dir_name = "multi_vars"  # Directory to save tracked variables
        self.vars = {}  # Dictionary to store tracked variables

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and precomputes necessary values for each variable.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        t_max = self.model.t_max  # Maximum simulation time
        dt = self.model.dt  # Time step size

        # Initialize storage for each variable to be tracked
        for var_ in self.var_list:
            self.vars[var_] = np.zeros(int(t_max / dt) + 1)

    def track(self):
        """
        Tracks and stores the values of each specified variable at each time step.

        This method should be called at each time step of the simulation.
        """
        step = self.model.step  # Current simulation step

        # Track the value of each variable at the specified cell index
        for var_ in self.var_list:
            self.vars[var_][step] = self.model.__dict__[var_][self.cell_ind[0], self.cell_ind[1]]

    def write(self):
        """
        Saves the tracked variables to disk as NumPy files.
        """
        # Create the output directory if it does not exist
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)

        # Save each tracked variable to a file
        for var_ in self.var_list:
            np.save(os.path.join(self.dir_name, var_), self.vars[var_])
