import numpy as np
from numba import njit
import json

from finitewave.core.tracker.tracker import Tracker


@njit
def _track_detectors_period(periods, detectors, detectors_state, u, t, threshold, step):
    """
    Numba-optimized function to track the activation periods of cells in a 2D mesh.

    Parameters
    ----------
    periods : np.ndarray
        Array to store the activation times of the detectors.
    detectors : np.ndarray
        Binary array indicating the presence of detectors at specific cells.
    detectors_state : np.ndarray
        Binary array indicating the state of detectors (1 if below threshold, 0 if above).
    u : np.ndarray
        The current potential values of the cardiac tissue.
    t : float
        The current simulation time.
    threshold : float
        The threshold value above which an activation is detected.
    step : int
        The current index in the periods array to store new activations.

    Returns
    -------
    periods : np.ndarray
        Updated periods array with new activation times.
    detectors_state : np.ndarray
        Updated state of detectors.
    step : int
        Updated step index after adding new activations.
    """
    n_i, n_j = u.shape
    for i in range(n_i):
        for j in range(n_j):
            if detectors[i, j] and u[i, j] > threshold and detectors_state[i, j]:
                periods[step] = [i, j, t]
                detectors_state[i, j] = 0
                step += 1
            elif detectors[i, j] and u[i, j] <= threshold and not detectors_state[i, j]:
                detectors_state[i, j] = 1

    return periods, detectors_state, step


class Period2DTracker(Tracker):
    """
    A class to track activation periods of cells in a 2D cardiac tissue model using detectors.

    Attributes
    ----------
    detectors : np.ndarray
        Binary array indicating the placement of detectors on the mesh.
    threshold : float
        The threshold potential value for detecting activations.
    _periods : np.ndarray
        Array to store the activation times for each detector.
    _detectors_state : np.ndarray
        Binary array indicating the state of detectors (1 if below threshold, 0 if above).
    _step : int
        The current index for storing activation periods.
    file_name : str
        The file name to save the tracked activation periods.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model and preallocates memory for tracking.
    track():
        Tracks the activation periods at each time step of the simulation.
    compute_periods():
        Computes the time intervals between successive activations for each detector.
    output():
        Property to get the computed activation periods.
    write():
        Saves the computed activation periods to a JSON file.
    """

    def __init__(self):
        """
        Initializes the Period2DTracker with default parameters.
        """
        Tracker.__init__(self)

        self.detectors = np.array([])  # Binary array indicating detector placement
        self.threshold = -40  # Threshold potential value for activation detection

        self._periods = np.array([])  # Array to store activation times
        self._detectors_state = np.array([])  # Array to store the state of each detector
        self._step = 0  # Current index in the periods array

        self.file_name = "period"  # File name for saving tracked data

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and preallocates memory for tracking.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model

        t_max = model.t_max  # Maximum simulation time
        dt = model.dt  # Time step size

        # Initial length of the periods array, scaled by the number of detectors
        n = 20 * len(self.detectors[self.detectors == 1])
        self._periods = -1 * np.ones([n, 3])
        self._detectors_state = np.ones(model.u.shape, dtype="uint8")

    def track(self):
        """
        Tracks the activation periods at each time step of the simulation.

        This method dynamically expands the periods array if necessary and updates
        the periods and detectors state arrays.
        """
        # Dynamically increase the size of the array if there is no free space
        if self._step == len(self._periods):
            self._periods = np.tile(self._periods, (2, 1))
            self._periods[len(self._periods) // 2:, :] = -1.

        # Update the periods and detectors state using the Numba-optimized function
        self._periods, self._detectors_state, self._step = _track_detectors_period(
            self._periods, self.detectors, self._detectors_state,
            self.model.u, self.model.t, self.threshold, self._step)

    def compute_periods(self):
        """
        Computes the time intervals between successive activations for each detector.

        Returns
        -------
        periods_dict : dict
            A dictionary where each key is a detector's coordinates and each value is a list of activation times and periods.
        """
        periods_dict = dict()
        to_str = lambda i, j: str(int(i)) + "," + str(int(j))

        # Iterate over the recorded periods and group them by detector coordinates
        for i in range(len(self._periods)):
            if self._periods[i][0] < 0:
                continue
            key = to_str(*self._periods[i][:2])
            if key not in periods_dict:
                periods_dict[key] = []
            periods_dict[key].append(self._periods[i][2])

        # Calculate the time intervals between successive activations
        for key in periods_dict:
            time_per_list = []
            for i, t in enumerate(periods_dict[key]):
                if i == 0:
                    time_per_list.append([t, 0])
                else:
                    time_per_list.append([t, t - time_per_list[i - 1][0]])
            periods_dict[key] = time_per_list

        return periods_dict

    @property
    def output(self):
        """
        Property to get the computed activation periods.
        """
        return self.compute_periods()

    def write(self):
        """
        Saves the computed activation periods to a JSON file.
        """
        jdata = json.dumps(self.compute_periods())
        with open(os.path.join(self.path, self.file_name), "w") as jf:
            jf.write(jdata)
