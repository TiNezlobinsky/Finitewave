from pathlib import Path
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    variable_name : str
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
        self.dir_name = "animation"   # Directory for saving frames
        self.variable_name = ""       # Name of the target array to capture
        self.frame_type = "float64"   # Default frame format settings
        self._frame_counter = 0       # Internal frame counter

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and sets up directories for saving frames.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        self._frame_counter = 0  # Reset frame counter

        if not Path(self.path, self.dir_name).is_dir():
            Path(self.path, self.dir_name).mkdir(parents=True)

    def _track(self):
        """
        Saves frames based on the specified step interval and target array.

        The frames are saved in the specified directory as NumPy files.
        """
        frame = self.model.__dict__[self.variable_name]
        dir_path = Path(self.path, self.dir_name)

        np.save(dir_path.joinpath(str(self._frame_counter)
                                  ).with_suffix(".npy"),
                frame.astype(self.frame_type))

        self._frame_counter += 1

    def write(self, scale=1, fps=30, cmap="viridis", clim=[0, 1],
              codec='mp4v', clear=False):
        """
        No operation. Exists to fulfill the interface requirements.

        Parameters
        ----------
        scale : int
            The scaling factor for the frames. Default is 1.
        fps : int
            The frames per second for the output video. Default is 30.
        cmap : str
            The colormap to use for plotting the frames.
        clim : list
            The color limits for the colormap.
        codec : str
            The codec to use for the output video. Default is 'mp4v'.
        clear : bool
            Whether to clear the figure before plotting the next frame.
        """
        path = Path(self.path)
        path_save = path.joinpath(self.dir_name).with_suffix(".mp4")

        height, width = np.array(self.model.u.shape) * scale
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(path_save, fourcc, fps, (width, height))

        cmap = plt.get_cmap(cmap)

        for i in range(self._frame_counter):
            frame = np.load(path.joinpath(str(i)).with_suffix(".npy"))
            # Normalize the frame data to the colormap
            frame = np.clip(frame, clim[0], clim[1])
            frame = (frame - clim[0]) / (clim[1] - clim[0])

            # Upscale the frame if necessary
            if scale > 1:
                frame = np.repeat(np.repeat(frame, scale, axis=0), scale,
                                  axis=1)
            # Convert the frame to an 8-bit RGB image
            frame_rgb = (cmap(frame) * 255).astype(np.uint8)
            out.write(frame_rgb)

        # Release everything when done
        out.release()

        if clear:
            shutil.rmtree(path.joinpath(self.dir_name))
            self._frame_counter = 0
