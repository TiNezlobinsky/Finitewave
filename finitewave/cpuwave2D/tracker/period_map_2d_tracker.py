import os
import numpy as np

from finitewave.cpuwave2D.tracker.animation_2d_tracker import Animation2DTracker


class PeriodMap2DTracker(Animation2DTracker):
    """
    A class to track the periods of activation for each cell in a 2D cardiac tissue model.

    This class extends Animation2DTracker to create and save a period map that shows the time interval between
    successive activations of each cell that crosses a given threshold. The period map is saved at each time step.

    Attributes
    ----------
    threshold : float
        The threshold potential value for detecting activations.
    period_map : np.ndarray
        2D array to store the time interval between successive activations for each cell.
    _period_map_state : np.ndarray
        2D array to store the current state of each cell (1 if below threshold, 0 if above).
    _last_time_map : np.ndarray
        2D array to store the last activation time for each cell.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model and preallocates memory for tracking.
    track():
        Tracks the activation periods at each time step of the simulation and saves them to files.
    write():
        Overridden method to handle file writing, here it's empty.
    """

    def __init__(self):
        """
        Initializes the PeriodMap2DTracker with default parameters.
        """
        Animation2DTracker.__init__(self)

        self.dir_name = "period"  # Directory to save the period maps

        self.threshold = -40.  # Threshold potential value for activation detection
        self.period_map = np.array([])  # Array to store activation periods
        self._period_map_state = np.array([])  # Array to store state of each cell for activation tracking

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and preallocates memory for tracking.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        # Call the parent class initialization method
        Animation2DTracker.initialize(self, model)

        # Initialize the period map and state arrays
        self.period_map = -1 * np.ones(self.model.u.shape)
        self._last_time_map = -1 * np.ones(self.model.u.shape)
        self._period_map_state = np.ones(self.model.u.shape, dtype="uint8")

    def track(self):
        """
        Tracks the activation periods at each time step of the simulation.

        This method calculates the time interval between successive activations for each cell,
        updates the period map, and saves it to a file.
        """
        # Check if the time to save a frame has been reached
        if self._t > self.step:
            # Identify active nodes where the state is 1 and the potential exceeds the threshold
            active_nodes = np.logical_and(self._period_map_state == 1, self.model.u > self.threshold)
            
            # Update the period map with the time interval between successive activations
            self.period_map[active_nodes] = self.model.t - self._last_time_map[active_nodes]
            
            # Update the last activation time for active nodes
            self._last_time_map[active_nodes] = self.model.t
            
            # Update the state of the nodes based on their potential values
            self._period_map_state[active_nodes] = 0
            self._period_map_state[np.logical_and(self._period_map_state == 0, self.model.u < self.threshold)] = 1

            # Save the current period map to a file
            np.save(os.path.join(self.path, self.dir_name, str(self._frame_n)), self.period_map)
            self._frame_n += 1  # Increment the frame counter
            self._t = 0  # Reset the time counter
        else:
            # Increment the time counter if the frame interval has not been reached
            self._t += self._dt

    def write(self):
        """
        Overridden write method.

        This method is intentionally left empty because the write functionality is handled in the track method.
        """
        pass
