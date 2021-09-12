from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import copy
import os


class CardiacModel:
    """Base class for electrophysiological models.

    Attributes
    ----------
    u : ndarray (or compatible type)
        Action potentital array (mV).

    dt : float
        Time step.

    dr : float
        Spatial step.

    t_max : float
        Maximum calculation time (model units).

    t : float
        Current time (model units).

    step : int
        Current step (iteration) of the calculation.

    """
    __metaclass__ = ABCMeta

    def __init__(self):
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
        pass

    @abstractmethod
    def diffuse_kernel(u_new, u, w, mesh):
        pass

    @abstractmethod
    def save_state(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @abstractmethod
    def load_state(self, path):
        pass

    def initialize(self):
        shape = self.cardiac_tissue.mesh.shape
        self.u = np.zeros(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.cardiac_tissue.compute_weights(self.dr, self.dt)
        self.cardiac_tissue.set_dtype(self.npfloat)

        if self.stim_sequence:
            self.stim_sequence.initialize(self)
        if self.tracker_sequence:
            self.tracker_sequence.initialize(self)
        if self.command_sequence:
            self.command_sequence.initialize(self)

        if self.state_keeper and self.state_keeper.record_load:
            self.state_keeper.load(self)

    def run(self, initialize=True):
        if initialize:
            self.initialize()

        if self.prog_bar:
            pbar = tqdm(total=int(np.ceil(self.t_max / self.dt)))

        while self.step < np.ceil(self.t_max / self.dt):
            if self.stim_sequence:
                self.stim_sequence.stimulate_next()

            self.run_diffuse_kernel()

            if self.tracker_sequence:
                self.tracker_sequence.tracker_next()

            self.run_ionic_kernel()

            self.t += self.dt
            self.step += 1
            self.u_new, self.u = self.u, self.u_new

            if self.command_sequence:
                self.command_sequence.execute_next()

            if pbar:
                pbar.update()
        if pbar:
            pbar.close()

        if self.state_keeper and self.state_keeper.record_save:
            self.state_keeper.save(self)

    def run_diffuse_kernel(self):
        self.diffuse_kernel(self.u_new, self.u, self.cardiac_tissue.weights,
                            self.cardiac_tissue.mesh)

    def clone(self):
        return copy.deepcopy(self)
