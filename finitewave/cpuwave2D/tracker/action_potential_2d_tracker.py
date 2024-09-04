import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class ActionPotential2DTracker(Tracker):
    """
    A class to track and record the action potential of a specific cell in a 2D cardiac tissue model.

    This tracker monitors the membrane potential of a single cell at each time step and stores the data
    in an array for later analysis or visualization.

    Attributes
    ----------
    act_pot : np.ndarray
        Array to store the action potential values at each time step.
    cell_ind : list of int
        Coordinates of the cell to be tracked in the 2D model grid.
    file_name : str
        Name of the file where the tracked action potential data will be saved.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model, setting up the action potential array.
    track():
        Records the action potential of the specified cell at the current time step.
    output():
        Returns the tracked action potential data.
    write():
        Saves the tracked action potential data to a file.
    """

    def __init__(self):
        """
        Initializes the ActionPotential2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.act_pot = np.array([])  # Initialize the array to store action potential
        self.cell_ind = [1, 1]       # Default cell indices to track
        self.file_name = "act_pot"   # Default file name for saving data

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model, setting up the action potential array.

        Parameters
        ----------
        model : object
            The cardiac tissue model object that contains simulation parameters like `t_max` (maximum time)
            and `dt` (time step).
        """
        self.model = model
        t_max = self.model.t_max  # Maximum simulation time
        dt = self.model.dt        # Time step
        self.act_pot = np.zeros(int(t_max / dt) + 1)  # Initialize the action potential array

    def track(self):
        """
        Records the action potential of the specified cell at the current time step.

        The action potential value is retrieved from the model's `u` matrix at the coordinates specified
        by `cell_ind`.
        """
        step = self.model.step  # Current simulation step
        # Record the action potential value at the specified cell index
        self.act_pot[step] = self.model.u[self.cell_ind[0], self.cell_ind[1]]

    @property
    def output(self):
        """
        Returns the tracked action potential data.

        Returns
        -------
        np.ndarray
            The array containing the tracked action potential values.
        """
        return self.act_pot

    def write(self):
        """
        Saves the tracked action potential data to a file.

        The file is saved in the path specified by `self.path` with the name `self.file_name`.
        """
        np.save(os.path.join(self.path, self.file_name), self.act_pot)
