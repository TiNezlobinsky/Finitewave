from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import copy


class CardiacModel(metaclass=ABCMeta):
    """
    Base class for electrophysiological models.

    This class serves as the base for implementing various cardiac models. It provides methods for
    initializing the model, running simulations, and managing the state of the simulation.

    Attributes
    ----------
    cardiac_tissue : CardiacTissue
        The tissue object that represents the cardiac tissue in the simulation.

    stim_sequence : StimSequence
        The sequence of stimuli applied to the cardiac tissue.

    tracker_sequence : TrackerSequence
        The sequence of trackers used to monitor the simulation.

    command_sequence : CommandSequence
        The sequence of commands to execute during the simulation.

    state_keeper : StateKeeper
        The object responsible for saving and loading the state of the simulation.

    stencil : Stencil
        The stencil used for numerical computations.

    u : ndarray
        Array representing the action potential (mV) across the tissue.

    u_new : ndarray
        Array for storing the updated action potential values.

    dt : float
        Time step for the simulation.

    dr : float
        Spatial step for the simulation.

    t_max : float
        Maximum time for the simulation (model units).

    t : float
        Current time in the simulation (model units).

    step : int
        Current step or iteration in the simulation.

    npfloat : str
        Data type used for floating-point operations. Default is ``float64``.

    prog_bar : bool
        Flag to enable or disable the progress bar during simulation.

    state_vars : list
        List of state variables to be saved and restored.
    """

    # __metaclass__ = ABCMeta

    def __init__(self):
        """
        Initializes the CardiacModel instance with default parameters and attributes.
        """
        self.cardiac_tissue = None
        self.stim_sequence = None
        self.tracker_sequence = None
        self.command_sequence = None
        self.state_keeper = None
        self.stencil = None

        self.diffuse_kernel = None
        self.ionic_kernel = None

        self.u = np.ndarray
        self.u_new = np.ndarray
        self.dt = 0.
        self.dr = 0.
        self.t_max = 0.
        self.t = 0
        self.step = 0
        
        self.npfloat = 'float64'
        self.prog_bar = True
        self.state_vars = []

    @abstractmethod
    def run_ionic_kernel(self):
        """
        Abstract method for running the ionic kernel. Must be implemented by
        subclasses.
        """
        pass

    def initialize(self):
        """
        Initializes the model for simulation. Sets up arrays, computes weights,
        and initializes stimuli, trackers, and commands.
        """
        shape = self.cardiac_tissue.mesh.shape
        self.u = np.zeros(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.cardiac_tissue.compute_weights(self.dr, self.dt)
        self.cardiac_tissue.set_dtype(self.npfloat)

        self.step = 0
        self.t = 0

        if self.stim_sequence:
            self.stim_sequence.initialize(self)

        if self.tracker_sequence:
            self.tracker_sequence.initialize(self)

        if self.command_sequence:
            self.command_sequence.initialize(self)

        if self.state_keeper:
            self.state_keeper.initialize(self)

    def run(self, initialize=True):
        """
        Runs the simulation loop. Handles stimuli, diffusion, ionic kernel
        updates, and tracking.

        Parameters
        ----------
        initialize : bool, optional
            Whether to (re)initialize the model before running the simulation.
            Default is True.
        """
        if initialize:
            self.initialize()

        # while self.step < np.ceil(self.t_max / self.dt):
        iters = int(np.ceil(self.t_max / self.dt))
        bar_desc = f"Running {self.__class__.__name__}"

        for _ in tqdm(range(iters), total=iters, desc=bar_desc,
                      disable=not self.prog_bar):

            if self.state_keeper and self.state_keeper.record_load:
                self.state_keeper.load()

            if self.stim_sequence:
                self.stim_sequence.stimulate_next()

            self.run_diffuse_kernel()
            self.transmembrane_current = self.u_new - self.u
            self.run_ionic_kernel()

            if self.tracker_sequence:
                self.tracker_sequence.tracker_next()

            self.t += self.dt
            self.step += 1
            self.u_new, self.u = self.u, self.u_new

            if self.command_sequence:
                self.command_sequence.execute_next()

            if self.check_termination():
                break

        if self.state_keeper and self.state_keeper.record_save:
            self.state_keeper.save()

    def check_termination(self):
        """
        Checks whether the simulation should terminate based on the current
        time. The ``CommandSequence`` may change the ``t_max`` value during
        execution to control the simulation duration.

        Returns
        -------
        bool
            True if the simulation should terminate, False otherwise.
        """
        max_iters = int(np.ceil(self.t_max / self.dt))
        return (self.t >= self.t_max) or (self.step >= max_iters)

    def run_diffuse_kernel(self):
        """
        Executes the diffusion kernel computation using the current parameters and tissue weights.
        """
        self.diffuse_kernel(self.u_new, self.u, self.cardiac_tissue.weights,
                            self.cardiac_tissue.mesh)

    def clone(self):
        """
        Creates a deep copy of the current model instance.

        Returns
        -------
        CardiacModel
            A deep copy of the current CardiacModel instance.
        """
        return copy.deepcopy(self)
