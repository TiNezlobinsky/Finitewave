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
    duration : float
        The duration for which the stimulation will be applied.
    passed : bool
        A flag indicating whether the stimulation has been applied.
    """

    def __init__(self, time, duration=0.0, selection=None):
        """
        Initializes the Stim object with the specified time.

        Parameters
        ----------
        time : float
            The time at which the stimulation is scheduled to occur.
        duration : float, optional
            The duration for which the stimulation will be applied. The default
            value is 0.0, indicating that the stimulation will be applied
            instantaneously.
        selection : object, optional
            A mesh-specific selector whose ``select(tissue)`` method returns
            local indexes in the model state vector.
        """
        self.t = time
        self.duration = duration
        self.passed = False
        self.selection = selection
        self.stim_area = None

    @abstractmethod
    def stimulate(self, simulation):
        """
        Applies the stimulation to the provided model.
        """
        pass

    def initialize(self, simulation):
        """
        Prepares the stimulation for application.
        """
        self.simulation = simulation
        if self.selection is not None:
            indexes = self.selection.select(simulation.cardiac_tissue)
            self.stim_indexes = simulation.backend.wrap_indexes(indexes)
        else:
            self.stim_indexes = self.stim_area.build_stim_indexes(simulation)

    def update_status(self, simulation):
        """
        Marks the stimulation as completed.
        """
        duration = max(self.duration, simulation.dt / 2)
        self.passed = ((simulation.t > (self.t + duration)) |
                       (simulation.t < self.t - simulation.dt / 2))
        return self.passed
