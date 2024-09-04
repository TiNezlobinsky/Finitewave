import os
import numpy as np
from numba import njit, prange
from scipy.spatial import distance

from finitewave.core.tracker.tracker import Tracker


class ECG2DTracker(Tracker):
    """
    A class to compute and track electrocardiogram (ECG) signals from a 2D cardiac tissue model simulation.

    This tracker calculates ECG signals at specified measurement points by computing the potential differences
    across the cardiac tissue mesh and considering the inverse square of the distance from each measurement point.

    Attributes
    ----------
    measure_points : np.ndarray
        An array of points (x, y, z) where ECG signals are measured.
    ecg : np.ndarray
        The computed ECG signals.
    step : int
        Interval in time steps at which ECG signals are calculated.
    _index : int
        Internal counter to keep track of the current step index for saving ECG signals.
    tissue_points : tuple
        Indices of the tissue points in the cardiac mesh where the potential is measured.
    distances : np.ndarray
        Precomputed squared distances between measurement points and tissue points.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model and precomputes necessary values.
    calc_ecg():
        Calculates the ECG signal based on the current potential difference in the model.
    track():
        Tracks and stores ECG signals at the specified intervals.
    write():
        Saves the computed ECG signals to disk as a NumPy file.
    """

    def __init__(self):
        """
        Initializes the ECG2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.measure_points = np.array([[0, 0, 1]])  # Default measurement points
        self.ecg = np.ndarray  # Placeholder for ECG data array
        self.step = 1  # Interval for ECG calculation
        self._index = 0  # Internal step counter

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and precomputes necessary values.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        n = int(np.ceil(model.t_max / (self.step * model.dt)))  # Number of steps to save ECG data
        self.ecg = np.zeros((self.measure_points.shape[0], n))  # Initialize ECG array

        # Get the cardiac tissue mesh and find tissue points
        mesh = model.cardiac_tissue.mesh
        self.tissue_points = np.where(mesh == 1)

        # Calculate distances from measure points to each tissue point
        points = np.argwhere(mesh == 1)
        tissue_points = np.append(points, np.zeros((points.shape[0], 1)), axis=1)  # Add zero z-coordinate
        self.distances = distance.cdist(self.measure_points, tissue_points)  # Compute distances
        self.distances = self.distances**2  # Square distances for inverse-square law

    def calc_ecg(self):
        """
        Calculates the ECG signal based on the current potential difference in the model.

        Returns
        -------
        np.ndarray
            The calculated ECG signals for each measurement point.
        """
        # Compute the current potential difference across the tissue points
        current = (self.model.u_new - self.model.u)[self.tissue_points]
        # Calculate the ECG signal by summing the contributions weighted by the inverse squared distances
        return np.sum(current / self.distances, axis=1)

    def track(self):
        """
        Tracks and stores ECG signals at the specified intervals.

        This method should be called at each time step of the simulation.
        """
        # Only compute ECG if the current step is a multiple of the step interval
        if self.model.step % self.step == 0:
            self.ecg[:, self._index] = self.calc_ecg()  # Calculate and store ECG
            self._index += 1  # Increment the step index

    def write(self):
        """
        Saves the computed ECG signals to disk as a NumPy file.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        # Save ECG data to a file
        np.save(os.path.join(self.dir_name, "ecg.npy"), self.ecg)
