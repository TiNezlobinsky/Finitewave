from pathlib import Path
import numpy as np
from scipy.spatial import distance

from finitewave.core.tracker.tracker import Tracker


class ECG2DTracker(Tracker):
    """
    A class to compute and track electrocardiogram (ECG) signals from a 2D
    cardiac tissue model simulation.

    This tracker calculates ECG signals at specified measurement points by
    computing the potential differences across the cardiac tissue mesh and
    considering the inverse square of the distance from each measurement point.

    Attributes
    ----------
    measure_points : np.ndarray
        An array of points (x, y, z) where ECG signals are measured.
    ecg : list
        The computed ECG signals.
    file_name : str
        The name of the file to save the computed ECG signals.
    distances : np.ndarray
        Precomputed squared distances between measurement points and tissue
        points.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model and precomputes
        necessary values.
    _track():
        Tracks and stores ECG signals at the specified intervals.
    output():
        Returns the computed ECG signals as a NumPy array.
    write():
        Saves the computed ECG signals to disk as a NumPy file.
    """

    def __init__(self):
        """
        Initializes the ECG2DTracker with default parameters.
        """
        super().__init__()
        self.measure_points = [1, 1, 1]  # Default measurement points
        self.ecg = []                 # Placeholder for ECG data array
        self.file_name = "ecg.npy"    # Default file name for saving ECG data

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and precomputes
        necessary values.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        self.ecg = []

        mesh = self.model.cardiac_tissue.mesh
        self.tissue_mask = mesh == 1
        coords = np.argwhere(self.tissue_mask)
        coords = np.column_stack((coords, np.zeros((len(coords), 1))))
        measure_points = np.atleast_2d(self.measure_points)
        self.distances = distance.cdist(measure_points, coords)

    def _track(self):
        """
        Tracks and stores ECG signals at the specified intervals.

        This method should be called at each time step of the simulation.
        """
        current = self.model.transmembrane_current[self.tissue_mask]
        ecg = np.sum(current / self.distances, axis=1)
        self.ecg.append(ecg)

    @property
    def output(self):
        """
        Returns the computed ECG signals as a NumPy array.
        """
        return np.squeeze(self.ecg)

    def write(self):
        """
        Saves the computed ECG signals to disk as a NumPy file.
        """
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True)

        np.save(Path(self.path).joinpath(self.file_name).with_suffix('.npy'),
                self.output)
