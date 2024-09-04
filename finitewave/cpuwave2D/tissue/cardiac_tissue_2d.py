import numpy as np

from finitewave.core.tissue.cardiac_tissue import CardiacTissue
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import IsotropicStencil2D


class CardiacTissue2D(CardiacTissue):
    """
    A class to represent a 2D cardiac tissue model with isotropic or anisotropic properties.

    Inherits from:
    -----------
    CardiacTissue
        Base class for cardiac tissue models.

    Attributes
    ----------
    shape : tuple of int
        Shape of the 2D grid for the cardiac tissue.
    mesh : np.ndarray
        Grid representing the tissue, with boundaries set to zero.
    stencil : IsotropicStencil2D
        Stencil for calculating weights in the 2D grid.
    conductivity : float
        Conductivity value for the tissue.
    fibers : np.ndarray or None
        Array representing fiber orientations. If None, isotropic weights are used.
    meta : dict
        Metadata about the tissue, including dimensionality.
    weights : np.ndarray
        Weights used for diffusion calculations.

    Methods
    -------
    __init__(shape):
        Initializes the 2D cardiac tissue model with the given shape and mode.
    add_boundaries():
        Sets boundary values in the mesh to zero.
    compute_weights(dr, dt):
        Computes the weights for diffusion based on the stencil and mode.
    """

    def __init__(self, shape):
        """
        Initializes the CardiacTissue2D model.

        Parameters
        ----------
        shape : tuple of int
            Shape of the 2D grid for the cardiac tissue.
        """
        CardiacTissue.__init__(self)
        self.meta["Dim"] = 2
        self.shape = shape
        self.mesh = np.ones(shape)
        self.add_boundaries()
        self.stencil = IsotropicStencil2D()
        self.conductivity = 1
        self.fibers = None

    def add_boundaries(self):
        """
        Sets the boundary values of the mesh to zero.

        The boundaries are defined as the edges of the grid, and this method
        updates these edges in the mesh array.
        """
        self.mesh[0, :] = 0
        self.mesh[:, 0] = 0
        self.mesh[-1, :] = 0
        self.mesh[:, -1] = 0

    def compute_weights(self, dr, dt):
        """
        Computes the weights for diffusion using the stencil and given parameters.

        Parameters
        ----------
        dr : float
            Spatial resolution.
        dt : float
            Temporal resolution.
        """
        self.weights = self.stencil.get_weights(self.mesh, self.conductivity,
                                                self.fibers, self.D_al,
                                                self.D_ac, dt, dr)
