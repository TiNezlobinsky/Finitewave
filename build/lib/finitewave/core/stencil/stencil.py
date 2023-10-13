from abc import ABCMeta, abstractmethod
from collections import defaultdict


class Stencil:
    """Base class for calculating stencil weights.

    Attributes
    ----------
    cache: dictionary
        Cache to reduce number of symbolic calculations

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.cache = defaultdict()

    @abstractmethod
    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        """Return weights as np.ndarray

        Arguments
        ---------
        mesh : np.ndarray
            Tissue array

        conductivity: np.ndarray or const
        The coefficient for imitating low conductance (fibrosis) areas

        fibers: np.ndarray
            Fibers orientation vectors

        D_al : float
            Diffusion along the fibers direction.

        D_ac : float
            Diffusion across the fibers direction.

        dt: float
            Time step

        dr: float
            Spatial step
        """
        pass
