from abc import ABCMeta, abstractmethod
import copy


class Tracker:
    """Base class for trackers used in simulations.

    This class provides a base implementation for trackers that monitor and record various aspects of the
    simulation. Trackers can be used to gather data such as activation times, wave dynamics, or ECG readings.

    Attributes
    ----------
    model : CardiacModel
        The simulation model to which the tracker is attached. This allows the tracker to access the model's state
        and data during the simulation.
    
    file_name : str
        The name of the file where the tracked data will be saved. Default is an empty string.
    
    path : str
        The directory path where the tracked data will be saved. Default is the current directory.

    Methods
    -------
    initialize(model)
        Abstract method to be implemented by subclasses for initializing the tracker with the simulation model.

    track()
        Abstract method to be implemented by subclasses for tracking and recording data during the simulation.

    clone()
        Creates a deep copy of the current tracker instance.

    write()
        Abstract method to be implemented by subclasses for writing the tracked data to a file.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Initializes the Tracker instance with default attributes.
        """
        self.model = None
        self.file_name = ""
        self.path = "."

    @abstractmethod
    def initialize(self, model):
        """
        Abstract method to be implemented by subclasses for initializing the tracker with the simulation model.

        Parameters
        ----------
        model : CardiacModel
            The simulation model to which the tracker will be attached.
        """
        pass

    @abstractmethod
    def track(self):
        """
        Abstract method to be implemented by subclasses for tracking and recording data during the simulation.
        """
        pass

    def clone(self):
        """
        Creates a deep copy of the current tracker instance.

        Returns
        -------
        Tracker
            A deep copy of the current Tracker instance.
        """
        return copy.deepcopy(self)

    @abstractmethod
    def write(self):
        """
        Abstract method to be implemented by subclasses for writing the tracked data to a file.
        """
        pass
