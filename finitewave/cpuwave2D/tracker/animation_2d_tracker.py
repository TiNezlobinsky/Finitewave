import os
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class Animation2DTracker(Tracker):
    """
    A class to track and save frames of a 2D cardiac tissue model simulation for animation purposes.

    This tracker periodically saves the state of a specified target array from the model to disk as NumPy files,
    which can later be used to create animations.

    Attributes
    ----------
    step : int
        Interval in time steps at which frames are saved.
    start : float
        The time at which to start recording frames.
    _t : float
        Internal counter for keeping track of the elapsed time since the last frame was saved.
    dir_name : str
        Directory name where animation frames are stored.
    _frame_n : int
        Internal counter to keep track of the number of frames saved.
    target_array : str
        The name of the model attribute to be saved as a frame.
    frame_format : dict
        A dictionary defining the format of saved frames. Contains 'type' (data type) and 'mult' (multiplier for scaling).
    _frame_format_type : str
        Internal storage for the data type of the saved frames.
    _frame_format_mult : float
        Internal storage for the multiplier for scaling the saved frames.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model and sets up directories for saving frames.
    track():
        Saves frames based on the specified step interval and target array.
    write():
        No operation. Exists to fulfill the interface requirements.
    """

    def __init__(self):
        """
        Initializes the Animation2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.step = 1  # Interval for frame capture
        self.start = 0  # Start time for capturing frames
        self._t = 0  # Internal time counter

        self.dir_name = "animation"  # Directory for saving frames

        self._frame_n = 0  # Frame counter

        self.target_array = ""  # Name of the target array to capture

        # Frame format: type (data type of saved frames), mult (scaling multiplier)
        self.frame_format = {
            "type": "float64",
            "mult": 1
        }

        self._frame_format_type = ""  # Internal storage for frame format type
        self._frame_format_mult = 1  # Internal storage for frame format multiplier

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and sets up directories for saving frames.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model

        self._t = 0  # Reset internal time counter
        self._frame_n = 0  # Reset frame counter
        self._dt = self.model.dt  # Time step size from the model
        self._step = self.step - self._dt  # Adjusted step for saving frames

        # Create the directory for saving frames if it doesn't exist
        if not os.path.exists(os.path.join(self.path, self.dir_name)):
            os.makedirs(os.path.join(self.path, self.dir_name))

        # Store frame format settings
        self._frame_format_type = self.frame_format["type"]
        self._frame_format_mult = self.frame_format["mult"]

    def track(self):
        """
        Saves frames based on the specified step interval and target array.

        The frames are saved in the specified directory as NumPy files.
        """
        # Only start tracking if the current model time is beyond the start time
        if not self.model.t >= self.start:
            return

        # Save a frame if enough time has elapsed since the last frame
        if self._t > self._step:
            # Retrieve the target array from the model and scale it
            frame = (self.model.__dict__[self.target_array] * self._frame_format_mult).astype(self._frame_format_type)
            # Save the frame as a NumPy file
            np.save(os.path.join(self.path, self.dir_name, str(self._frame_n)), frame)
            self._frame_n += 1  # Increment frame counter
            self._t = 0  # Reset internal time counter
        else:
            self._t += self._dt  # Increment internal time counter by the time step

    def write(self):
        """
        No operation for this tracker. Exists to fulfill the interface requirements.
        """
        pass
