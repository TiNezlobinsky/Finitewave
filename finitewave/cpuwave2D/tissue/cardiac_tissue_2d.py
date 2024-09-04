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
    __init__(shape, mode='iso'):
        Initializes the 2D cardiac tissue model with the given shape and mode.
    add_boundaries():
        Sets boundary values in the mesh to zero.
    compute_weights(dr, dt):
        Computes the weights for diffusion based on the stencil and mode.
    """

    def __init__(self, shape, mode='iso'):
        """
        Initializes the CardiacTissue2D model.

        Parameters
        ----------
        shape : tuple of int
            Shape of the 2D grid for the cardiac tissue.
        mode : str, optional
            Mode for weight calculation. 'iso' for isotropic, 'aniso' for anisotropic. Default is 'iso'.
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

        Notes
        -----
        If the mode is set to 'iso', isotropic weights are computed. If the mode is
        'aniso', the method for anisotropic weights is commented out. This method
        should be updated to handle anisotropic weights if needed.
        """
        self.weights = self.stencil.get_weights(self.mesh, self.conductivity,
                                                self.fibers, self.D_al,
                                                self.D_ac, dt, dr)
        # if self.mode == 'iso':
        #     self.weights = self.stencil.get_weights(self.mesh,
        #                                             D_al*self.conductivity,
        #                                             dt,
        #                                             dr)
        # elif self.mode == 'aniso':
        #     self.weights = self.aniso_weights(dr, dt, D_al, D_ac)
        # else:
        #     raise IncorrectWeightsModeError2D()

    # def aniso_weights(self, dr, dt, D_al, D_ac):
    #     """
    #     Computes the anisotropic weights for diffusion.
    #
    #     Parameters
    #     ----------
    #     dr : float
    #         Spatial resolution.
    #     dt : float
    #         Temporal resolution.
    #     D_al : float
    #         Longitudinal diffusion coefficient.
    #     D_ac : float
    #         Cross-sectional diffusion coefficient.
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         Array of weights for anisotropic diffusion.
    #     """
    #     indexes = list(range(9))
    #
    #     index_map = dict(zip(range(len(indexes)), indexes))
    #     weights = np.zeros((*self.shape, len(indexes)))
    #     self._compute_diffuse(D_al, D_ac)
    #
    #     for i in range(1, self.shape[0] - 1):
    #         for j in range(1, self.shape[1] - 1):
    #             mesh_local = self.mesh[i-1: i+2, j-1: j+2]
    #             mesh_local[mesh_local != 1] = 0
    #             mesh_local = mesh_local.astype('int')
    #
    #             empty_center = mesh_local[1, 1] != 1
    #             isolated_center = np.sum(mesh_local) < 2
    #             if empty_center or isolated_center:
    #                 continue
    #
    #             diffuse = self.diffuse[i-1: i+2, j-1: j+2, :]
    #             local_weights = self.stencil.get_weights(mesh_local, diffuse, dt,
    #                                                      dr).flatten()
    #
    #             for ind in range(weights.shape[2]):
    #                 weights[i, j, ind] = local_weights[index_map[ind]]
    #     return weights

    # def set_dtype(self, dtype):
    #     """
    #     Sets the data type of the weights and mesh.
    #
    #     Parameters
    #     ----------
    #     dtype : type
    #         Desired data type for the weights and mesh.
    #     """
    #     self.weights = self.weights.astype(dtype)
    #     self.mesh = self.mesh.astype(dtype)

    # def _compute_diffuse(self, D_al, D_ac):
    #     """
    #     Computes the diffusion coefficients based on fiber orientation.
    #
    #     Parameters
    #     ----------
    #     D_al : float
    #         Longitudinal diffusion coefficient.
    #     D_ac : float
    #         Cross-sectional diffusion coefficient.
    #     """
    #     self.diffuse[:, :, 0] = ((D_ac + (D_al - D_ac) *
    #                              self.fibers[:, :, 0]**2) * self.conductivity)
    #     self.diffuse[:, :, 1] = (0.5 * (D_al - D_ac) *
    #                              self.fibers[:, :, 0] * self.fibers[:, :, 1] *
    #                              self.conductivity)
    #     self.diffuse[:, :, 2] = ((D_ac + (D_al - D_ac) *
    #                              self.fibers[:, :, 1]**2) * self.conductivity)
    #
    #     fibers_x = self.fibers[:-1, :, :] + self.fibers[1:, :, :]
    #     fibers_x = fibers_x / np.linalg.norm(fibers_x, axis=2)[:, :, np.newaxis]
    #     fibers_y = self.fibers[:, :-1, :] + self.fibers[:, 1:, :]
    #     fibers_y = fibers_y / np.linalg.norm(fibers_y, axis=2)[:, :, np.newaxis]
