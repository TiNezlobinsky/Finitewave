from abc import ABCMeta, abstractmethod
import numpy as np
import copy


class CardiacTissue:
    """Base class for a model tissue.

    Attributes
    ----------
    mesh : numpy array (or compatible type)
        Numpy array of integers where:
        0 - empty points.
        1 - cardiomyocyte.
        2 - fibrosis.

    conductvity : numpy array (or const), default: 1 (normal conductivity)
        Numpy array of floats [0, 1]. The coefficient for imitating low
        conductance (fibrosis) areas. Conductivities is mulplied to
        diffusion coefficiens.

    fibers : numpy array (or compatible type)
        Numpy 2D or 3D array with 2 or 3 components in each node. Specify the
        fibers direction.

    D_al : float
        Diffusion along the fibers direction.

    D_ac : float
        Diffusion across the fibers direction.

    weights : numpy array (or compatible type)
        Calculated weights based on cardiomyocytes/empty nodes/fibrosis nodes.

    shape : list, tuple
        Mesh size [ni, nj] or [ni, nj, nk].

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.mesh = np.array([], dtype="int8")
        self.conductvity = np.array([])
        self.fibers = np.array([])
        self.D_al = 1
        self.D_ac = 1
        self.weights = np.array([])
        self.boundary = np.array([], dtype="int16")
        self.shape = []
        self.meta = dict()

    @abstractmethod
    def add_boundaries(self):
        pass

    @abstractmethod
    def compute_weights(self):
        pass

    def add_pattern(self, fibro_pattern):
        fibro_pattern.apply(self)

    def clean(self):
        # remove all the fibrosis points (= 2)
        self.mesh[self.mesh == 2] = 1

    def clone(self):
        return copy.deepcopy(self)

    def set_dtype(self, dtype):
        self.weights = self.weights.astype(dtype)
        self.mesh = self.mesh.astype(dtype)
