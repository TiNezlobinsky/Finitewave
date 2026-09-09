import copy
from typing import Literal
import numpy as np


class Simulation:
    """Base class for cardiac simulations.

    This class stores simulation components and defines their initialization
    order.

    Attributes
    ----------
    meta : dict
        Metadata about the simulation, such as its name and description.
    cardiac_tissue : CardiacTissue
        The cardiac tissue on which the simulation is run.
    stim_sequence : StimSequence
        A sequence of stimuli to be applied during the simulation.
    tracker_sequence : TrackerSequence
        A sequence of trackers to record data during the simulation.
    command_sequence : CommandSequence
        A sequence of commands to be executed during the simulation.
    state_loader : StateLoader
        A component to load the simulation state from a file.
    state_saver : StateSaver
        A component to save the simulation state to a file.
    cardiac_model : CardiacModel
        The cardiac electrophysiology model used in the simulation.
    spatial_discretization : SpatialDiscretization
        The spatial discretization method used in the simulation.
    time_integration : TimeIntegration
        The time integration method used in the simulation.
    backend : Backend
        The computational backend used for the simulation.
    dt : float
        The time step size for the simulation.
    t_max : float
        The maximum simulation time.
    t : float
        The current simulation time.
    iteration : int
        The current iteration number of the simulation loop.
    """
    def __init__(self):
        self.meta = {}
        self.cardiac_tissue = None
        self.stim_sequence = None
        self.tracker_sequence = None
        self.command_sequence = None
        self.state_loader = None
        self.state_saver = None
        self.cardiac_model = None
        self.spatial_discretization = None
        self.time_integration = None
        self.backend = None

        self.dt = None
        self.t_max = None
        self.t = 0
        self.iteration = 0

    def initialize(self):
        """Initialize the model and attached simulation components.

        Sets up arrays, computes weights,
        and initializes stimuli, trackers, and commands.

        Note
        ----
        The order of initialization is important. Later components may depend
        on earlier ones being initialized first.
        """
        self.iteration = 0
        self.t = 0
        self.cardiac_model.initialize(self)
        self.spatial_discretization.initialize(self)
        self.time_integration.initialize(self)

        if self.stim_sequence:
            self.stim_sequence.initialize(self)

        if self.tracker_sequence:
            self.tracker_sequence.initialize(self)

        if self.command_sequence:
            self.command_sequence.initialize(self)

        if self.state_loader:
            self.state_loader.initialize(self)

        if self.state_saver:
            self.state_saver.initialize(self)

    def run(self):
        """Run the simulation loop."""
        raise NotImplementedError

    def _remaining_steps(self):
        """Return the number of complete fixed-size steps before ``t_max``.

        Ratios sufficiently close to an integer are rounded to that integer to
        avoid losing a step to floating-point representation.
        """
        remaining = self.t_max - self.t
        if remaining <= 0:
            return 0

        ratio = remaining / self.dt
        nearest = round(ratio)
        if np.isclose(ratio, nearest, rtol=1e-12, atol=1e-12):
            return max(0, int(nearest))

        return max(0, int(np.floor(ratio)))

    def check_termination(self):
        """Check whether another complete time step can be performed.

        The simulation terminates at ``t_max`` or when the remaining duration
        is shorter than ``dt``. A ``CommandSequence`` may change ``t_max``
        during execution to control the simulation duration.

        Returns
        -------
        bool
            True if no complete step remains; otherwise, False.
        """
        return self._remaining_steps() == 0

    def clone(self):
        """Create a deep copy of this simulation.

        Returns
        -------
        Simulation
            A deep copy of this simulation instance.
        """
        return copy.deepcopy(self)
