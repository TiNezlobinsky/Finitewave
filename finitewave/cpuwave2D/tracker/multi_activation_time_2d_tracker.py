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
    step : int
        The frequency of tracking.

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
        self.act_t = []  # Initialize activation times as an empty array
        self.threshold = -40  # Activation threshold
        self.file_name = "multi_act_time_2d"  # Output file name
        self.step = 1           # Frequency of tracking
        self.start_time = 0     # Start time for tracking
        self.end_time = np.inf  # End time for tracking
        self._step_counter = 0         # Counter for tracking frequency

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and
        precomputes necessary values.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        # Initialize with a single layer of -1 (no activation)
        self.act_t = [-np.ones_like(self.model.u)]
        # Initially mark all boundary cells as activated
        self._activated = np.full(self.model.u.shape, 0, dtype=bool)

    def track(self):
        """
        Tracks and stores activation times for each cell in
        the model at each time step.

        This method should be called at each time step of the simulation.
        """
        if self.start_time > self.model.t or self.model.t > self.end_time:
            return
        # Check if the current time step is within the tracking frequency
        if self._step_counter % self.step != 0:
            self._step_counter += 1
            return
        # Mask for cells that crossed the threshold and are not activated yet
        cross_mask = ((self.model.u >= self.threshold)
                      & (self._activated == 0))
        self._activated = np.where(cross_mask, 1, self._activated)
        # Set inactive cells that backcrossed the threshold after activation
        backcross_mask = ((self.model.u < self.threshold)
                          & (self._activated == 1))
        self._activated = np.where(backcross_mask, 0, self._activated)
        # Check if there are already activated cells in the current
        # activation layer
        if np.any(self.act_t[-1][cross_mask] > -1):
            self.act_t.append(-np.ones(self.model.u.shape))
        # Update activation times where the threshold is crossed
        self.act_t[-1] = np.where(cross_mask, self.model.t, self.act_t[-1])
        self._step_counter += 1

    @property
    def output(self):
        """
        Returns the activation times.

        Returns
        -------
        list of np.ndarray
            A list where each element is an array storing activation times
            for each cell.
        """
        return self.act_t

    def write(self):
        """
        Saves the activation times to disk as a NumPy file.
        """
        # Save the activation times list to a file
        np.save(os.path.join(self.path, self.file_name), self.act_t)
