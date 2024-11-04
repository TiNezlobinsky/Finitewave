import numbers
import numpy as np

from finitewave.core.stencil.stencil import Stencil


class IsotropicStencil3D(Stencil):
    """
    A class to represent a 3D isotropic stencil for diffusion processes.
    """
    def __init__(self):
        """
        Initializes the IsotropicStencil3D with default settings.
        """
        Stencil.__init__(self)

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        """
        Computes the weights for diffusion on a 3D mesh using an isotropic stencil.

        Parameters
        ----------
        mesh : np.ndarray
            3D array representing the mesh grid of the tissue. Non-tissue areas are set to 0.
        conductivity : float
            Conductivity of the tissue, which scales the diffusion coefficient.
        fibers : np.ndarray
            Array representing fiber orientations. Not used in isotropic stencil but kept for consistency.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient. Not used in isotropic stencil but kept for consistency.
        dt : float
            Temporal resolution.
        dr : float
            Spatial resolution.

        Returns
        -------
        np.ndarray
            4D array of weights for diffusion, with the shape of (mesh.shape[0], mesh.shape[1], 7).
        
        Notes
        -----
        The method assumes isotropic diffusion where `D_al` is used as the diffusion coefficient.
        The weights are computed for four directions (up, right, down, left) and the central weight.
        Heterogeneity in the diffusion coefficients is handled by adjusting the weights based on
        differences in the diffusion coefficients along the rows and columns.
        """
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        weights = np.zeros((*mesh.shape, 7))

        diffuse = D_al * conductivity * np.ones(mesh.shape)

        weights[:, :, :, 0] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=0)
        weights[:, :, :, 1] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=1)
        weights[:, :, :, 2] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=2)
        weights[:, :, :, 4] = diffuse * dt / (dr**2) * np.roll(mesh, -1,
                                                               axis=2)
        weights[:, :, :, 5] = diffuse * dt / (dr**2) * np.roll(mesh, -1,
                                                               axis=1)
        weights[:, :, :, 6] = diffuse * dt / (dr**2) * np.roll(mesh, -1,
                                                               axis=0)

        # heterogeneity
        diff_i = np.roll(diffuse, 1, axis=0) - np.roll(diffuse, -1, axis=0)
        diff_j = np.roll(diffuse, 1, axis=1) - np.roll(diffuse, -1, axis=1)
        diff_k = np.roll(diffuse, 1, axis=2) - np.roll(diffuse, -1, axis=2)

        weights[:, :, :, 0] -= dt / (2*dr) * diff_i
        weights[:, :, :, 1] -= dt / (2*dr) * diff_j
        weights[:, :, :, 2] -= dt / (2*dr) * diff_k
        weights[:, :, :, 4] += dt / (2*dr) * diff_k
        weights[:, :, :, 5] += dt / (2*dr) * diff_j
        weights[:, :, :, 6] += dt / (2*dr) * diff_i

        for i in [0, 1, 2, 4, 5, 6]:
            weights[:, :, :, i] *= mesh
            weights[:, :, :, 3] -= weights[:, :, :, i]
        weights[:, :, :, 3] += 1
        weights[:, :, :, 3] *= mesh
        return weights
