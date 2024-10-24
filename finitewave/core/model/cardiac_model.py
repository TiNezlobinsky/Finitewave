from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import copy
import os


class CardiacModel:
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

    prog_bar : bool
        Flag to enable or disable the progress bar during simulation.

    state_vars : list
        List of state variables to be saved and restored.

    Methods
    -------
    run_ionic_kernel()
        Abstract method to be implemented by subclasses for running the ionic kernel.
    
    diffuse_kernel(u_new, u, w, mesh)
        Abstract method to be implemented by subclasses for diffusion computation.
    
    save_state(path)
        Abstract method to be implemented by subclasses for saving the simulation state.
    
    load_state(path)
        Abstract method to be implemented by subclasses for loading the simulation state.
    
    initialize()
        Initializes the model for simulation, setting up arrays and computing weights.
    
    run(initialize=True)
        Runs the simulation loop, handling stimuli, diffusion, ionic kernel updates, and tracking.
    
    run_diffuse_kernel()
        Runs the diffusion kernel computation.
    
    clone()
        Creates a deep copy of the current model instance.
    """

    __metaclass__ = ABCMeta

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

        self.u = np.ndarray
        self.u_new = np.ndarray
        self.dt = 0.
        self.dr = 0.
        self.t_max = 0.
        self.t = 0
        self.step = 0

        self.prog_bar = True
        self.state_vars = []

    @abstractmethod
    def run_ionic_kernel(self):
        """
        Abstract method for running the ionic kernel. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def diffuse_kernel(u_new, u, w, mesh):
        """
        Abstract method for diffusion computation. Must be implemented by subclasses.

        Parameters
        ----------
        u_new : ndarray
            The array to store updated action potential values.
        
        u : ndarray
            The current action potential array.
        
        w : ndarray
            The weights for the diffusion computation.
        
        mesh : ndarray
            The tissue mesh.
        """
        pass

    @abstractmethod
    def save_state(self, path):
        """
        Abstract method for saving the simulation state. Must be implemented by subclasses.

        Parameters
        ----------
        path : str
            The directory path where the state will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @abstractmethod
    def load_state(self, path):
        """
        Abstract method for loading the simulation state. Must be implemented by subclasses.

        Parameters
        ----------
        path : str
            The directory path from where the state will be loaded.
        """
        pass

    def initialize(self):
        """
        Initializes the model for simulation. Sets up arrays, computes weights, and initializes stimuli,
        trackers, and commands.
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

        if self.state_keeper and self.state_keeper.record_load:
            self.state_keeper.load(self)

    def run(self, initialize=True):
        """
        Runs the simulation loop. Handles stimuli, diffusion, ionic kernel updates, and tracking.

        Parameters
        ----------
        initialize : bool, optional
            Whether to (re)initialize the model before running the simulation. Default is True.
        """
        if initialize:
            self.initialize()

        # while self.step < np.ceil(self.t_max / self.dt):
        iters = int(np.ceil(self.t_max / self.dt))
        for _ in tqdm(range(iters), total=iters,
                      desc=f"Running {self.__class__.__name__}",
                      disable=not self.prog_bar):
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

        if self.state_keeper and self.state_keeper.record_save:
            self.state_keeper.save(self)

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
