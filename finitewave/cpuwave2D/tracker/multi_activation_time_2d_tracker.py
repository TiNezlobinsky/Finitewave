import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class MultiActivationTime2DTracker(Tracker):
    """
    A class to compute and track multiple activation times in a 2D cardiac tissue model simulation.

    This tracker monitors the potential across the cardiac tissue and records the times when cells surpass
    a specific threshold, supporting multiple activations such as re-entrant waves or multiple excitations.

    Attributes
    ----------
    act_t : list of np.ndarray
        A list where each element is an array storing activation times for each cell.
    threshold : float
        The potential threshold to determine cell activation.
    file_name : str
        The file name for saving the activation times.
    activated : np.ndarray
        A boolean array indicating whether each cell is currently activated.
    amount : np.ndarray
        An array storing the number of times each cell has been activated.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model and precomputes necessary values.
    track():
        Tracks and stores activation times for each cell in the model at each time step.
    output:
        Returns the activation times.
    write():
        Saves the activation times to disk as a NumPy file.
    """

    def __init__(self):
        """
        Initializes the MultiActivationTime2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.act_t = np.array([])  # Initialize activation times as an empty array
        self.threshold = -40  # Activation threshold
        self.file_name = "multi_act_time_2d"  # Output file name

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and precomputes necessary values.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        self.act_t = [-np.ones(self.model.u.shape)]  # Initialize with a single layer of -1 (no activation)
        self.activated = np.full(self.model.u.shape, True)  # Initially mark all boundary cells as activated
        self.activated[1:-1, 1:-1] = False  # Set internal cells as not activated
        self.amount = np.ones(self.model.u.shape)  # Track the number of activations for each cell

    def track(self):
        """
        Tracks and stores activation times for each cell in the model at each time step.

        This method should be called at each time step of the simulation.
        """
        # Calculate updated activation times where the threshold is crossed
        updated_array = np.where((self.act_t[-1] < 0) & (self.model.u > self.threshold), self.model.t, -1)

        # Update the amount of activations for cells that cross the threshold again
        if np.any((self.activated == False) & (self.act_t[-1] > 0) & (self.model.u > self.threshold)):
            self.amount = np.where(
                (self.activated == False) & (self.act_t[-1] > 0) & (self.model.u > self.threshold),
                self.amount + 1, self.amount
            )
            # If any cell has been activated more times than the current activation list length, append a new array
            if np.any(self.amount > len(self.act_t)):
                self.act_t.append(updated_array)
        else:
            # Update the last recorded activation times with new activations
            self.act_t[-1] = np.where(updated_array > 0, updated_array, self.act_t[-1])

        # Update the activation status of cells
        self.activated[1:-1, 1:-1] = np.where(
            (self.model.u[1:-1, 1:-1] > self.threshold) & (self.activated[1:-1, 1:-1] == False),
            True, self.activated[1:-1, 1:-1]
        )
        # Reset the activation status if the potential drops below the threshold
        self.activated[1:-1, 1:-1] = np.where(
            (self.model.u[1:-1, 1:-1] <= self.threshold) & (self.activated[1:-1, 1:-1] == True),
            False, self.activated[1:-1, 1:-1]
        )

    @property
    def output(self):
        """
        Returns the activation times.

        Returns
        -------
        list of np.ndarray
            A list where each element is an array storing activation times for each cell.
        """
        return self.act_t

    def write(self):
        """
        Saves the activation times to disk as a NumPy file.
        """
        # Save the activation times list to a file
        np.save(os.path.join(self.path, self.file_name), self.act_t)
