from abc import ABCMeta, abstractmethod
from collections import defaultdict

class Stencil:
    """Base class for calculating stencil weights used in numerical simulations.

    This abstract base class defines the interface for calculating stencil weights for numerical
    simulations. It includes a caching mechanism to optimize performance by reducing the number of
    symbolic calculations.

    Attributes
    ----------
    cache : dict
        A dictionary used to cache previously computed stencil weights to improve performance
        by avoiding redundant calculations.

    Methods
    -------
    get_weights(mesh, conductivity, fibers, D_al, D_ac, dt, dr)
        Abstract method that must be implemented by subclasses to compute and return stencil weights.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Initializes the Stencil object with an empty cache.
        """
        self.cache = defaultdict()

    @abstractmethod
    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        """
        Computes and returns the stencil weights based on the provided parameters.

        This method must be implemented by subclasses to compute the stencil weights used for
        numerical simulations. The weights are calculated based on the tissue mesh, conductivity,
        fibers orientation, diffusion coefficients, time step, and spatial step.

        Parameters
        ----------
        mesh : np.ndarray
            A 2D or 3D numpy array representing the tissue mesh where each value indicates the type
            of tissue (e.g., cardiomyocyte, fibrosis).

        conductivity : np.ndarray or float
            A numpy array or constant value representing the coefficient for imitating low conductance
            (fibrosis) areas. This affects the diffusion coefficients.

        fibers : np.ndarray
            A 2D or 3D numpy array representing the orientation vectors of the fibers within the tissue.

        D_al : float
            The diffusion coefficient along the fibers direction.

        D_ac : float
            The diffusion coefficient across the fibers direction.

        dt : float
            The time step used in the simulation.

        dr : float
            The spatial step used in the simulation.

        Returns
        -------
        np.ndarray
            A numpy array of stencil weights computed based on the provided parameters.
        """
        pass
