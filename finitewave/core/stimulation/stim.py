from abc import ABC, abstractmethod


class Stim(ABC):
    """Base class for stimulation in cardiac models.

    The ``Stim`` class represents a general stimulation object used in cardiac
    simulations. It provides methods to manage the timing and state of
    stimulation. Subclasses should implement specific stimulation behaviors.

    Attributes
    ----------
    t : float
        The time at which the stimulation is to occur.

    passed : bool
        A flag indicating whether the stimulation has been applied.
    """

    def __init__(self, time):
        """
        Initializes the Stim object with the specified time.

        Parameters
        ----------
        time : float
            The time at which the stimulation is scheduled to occur.
        """
        self.t = time
        self.passed = False

    @abstractmethod
    def stimulate(self, model):
        """
        Applies the stimulation to the provided model.
        """
        pass

    @abstractmethod
    def initialize(self, model):
        """
        Prepares the stimulation for application.
        """
        pass

    @abstractmethod
    def update_status(self, model):
        """
        Marks the stimulation as completed.
        """
        pass
