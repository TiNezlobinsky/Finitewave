import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


@njit
def coeffs(m0, m1, m2, m3):
    """
    Computes the coefficients used in the weight calculations.

    Parameters
    ----------
    m0 : float
        Mesh value at position (i-1, j-1).
    m1 : float
        Mesh value at position (i-1, j+1).
    m2 : float
        Mesh value at position (i, j-1).
    m3 : float
        Mesh value at position (i, j+1).

    Returns
    -------
    float
        Computed coefficient based on input values.
    """
    return m0 * m1 / (1 + m0 * m1 * m2 * m3)


@njit
def compute_weights(w, m, d_x, d_xy, d_yx, d_y):
    """
    Computes the weights for diffusion on a 2D mesh based on asymmetric stencil.

    Parameters
    ----------
    w : np.ndarray
        3D array to store the computed weights. Shape is (mesh.shape[0], mesh.shape[1], 9).
    m : np.ndarray
        2D array representing the mesh grid of the tissue.
    d_x : np.ndarray
        2D array with diffusion coefficients along the x-direction.
    d_xy : np.ndarray
        2D array with diffusion coefficients for cross-terms in x and y directions.
    d_y : np.ndarray
        2D array with diffusion coefficients along the y-direction.
    d_yx : np.ndarray
        2D array with diffusion coefficients for cross-terms in y and x directions.
    """
    n_i = m.shape[0]
    n_j = m.shape[1]
    for ii in prange(n_i * n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if m[i, j] != 1:
            continue

        w[i, j, 0] = 0.5 * (d_xy[i-1, j] * coeffs(m[i-1, j-1], m[i-1, j+1],
                                                  m[i, j-1], m[i, j+1]) +
                            d_yx[i, j-1] * coeffs(m[i-1, j-1], m[i+1, j-1],
                                                  m[i-1, j], m[i+1, j]))
        w[i, j, 1] = (d_x[i-1, j] * m[i-1, j] +
                      0.5 * (d_yx[i, j-1] * coeffs(m[i-1, j], m[i+1, j],
                                                   m[i-1, j-1], m[i+1, j-1]) -
                             d_yx[i, j] * coeffs(m[i-1, j], m[i+1, j],
                                                 m[i-1, j+1], m[i+1, j+1])))
        w[i, j, 2] = -0.5 * (d_xy[i-1, j] * coeffs(m[i-1, j-1], m[i-1, j+1],
                                                   m[i, j-1], m[i, j+1]) +
                             d_yx[i, j] * coeffs(m[i-1, j+1], m[i+1, j+1],
                                                 m[i-1, j], m[i+1, j]))
        w[i, j, 3] = (d_y[i, j-1] * m[i, j-1] +
                      0.5 * (d_xy[i-1, j] * coeffs(m[i, j-1], m[i, j+1],
                                                   m[i-1, j-1], m[i-1, j+1]) -
                             d_xy[i, j] * coeffs(m[i, j-1], m[i, j+1],
                                                 m[i+1, j-1], m[i+1, j+1])))
        w[i, j, 4] = - (m[i-1, j] * d_x[i-1, j] + m[i+1, j] * d_x[i, j] +
                        m[i, j-1] * d_y[i, j-1] + m[i, j+1] * d_y[i, j])
        w[i, j, 5] = (d_y[i, j] * m[i, j+1] +
                      0.5 * (-d_xy[i-1, j] * coeffs(m[i, j-1], m[i, j+1],
                                                    m[i-1, j-1], m[i-1, j+1]) +
                             d_xy[i, j] * coeffs(m[i, j-1], m[i, j+1],
                                                 m[i+1, j-1], m[i+1, j+1])))
        w[i, j, 6] = -0.5 * (d_xy[i, j] * coeffs(m[i+1, j-1], m[i+1, j+1],
                                                 m[i, j-1], m[i, j+1]) +
                             d_yx[i, j-1] * coeffs(m[i-1, j-1], m[i+1, j-1],
                                                   m[i-1, j], m[i+1, j]))
        w[i, j, 7] = (d_x[i, j] * m[i+1, j] +
                      0.5 * (-d_yx[i, j-1] * coeffs(m[i-1, j], m[i+1, j],
                                                    m[i-1, j-1], m[i+1, j-1]) +
                             d_yx[i, j] * coeffs(m[i-1, j], m[i+1, j],
                                                 m[i-1, j+1], m[i+1, j+1])))
        w[i, j, 8] = 0.5 * (d_xy[i, j] * coeffs(m[i+1, j-1], m[i+1, j+1],
                                                m[i, j-1], m[i, j+1]) +
                            d_yx[i, j] * coeffs(m[i-1, j+1], m[i+1, j+1],
                                                m[i-1, j], m[i+1, j]))


class AsymmetricStencil2D(Stencil):
    """
    A class to represent a 2D asymmetric stencil for diffusion processes.

    Inherits from:
    -----------
    Stencil
        Base class for different stencils used in diffusion calculations.

    Methods
    -------
    get_weights(mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        Computes the weights for diffusion based on the asymmetric stencil.
    """

    def __init__(self):
        """
        Initializes the AsymmetricStencil2D with default settings.
        """
        Stencil.__init__(self)

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        """
        Computes the weights for diffusion on a 2D mesh using an asymmetric stencil.

        Parameters
        ----------
        mesh : np.ndarray
            2D array representing the mesh grid of the tissue. Non-tissue areas are set to 0.
        conductivity : float
            Conductivity of the tissue, which scales the diffusion coefficient.
        fibers : np.ndarray
            Array representing fiber orientations. Used to compute directional diffusion coefficients.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.
        dt : float
            Temporal resolution.
        dr : float
            Spatial resolution.

        Returns
        -------
        np.ndarray
            3D array of weights for diffusion, with the shape of (mesh.shape[0], mesh.shape[1], 9).

        Notes
        -----
        The method assumes asymmetric diffusion where different coefficients are used for different directions.
        The weights are computed for eight surrounding directions and the central weight, based on the asymmetric stencil.
        Heterogeneity in the diffusion coefficients is handled by adjusting the weights based on fiber orientations.
        """
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        # fibers[np.where(mesh != 1)] = 0
        weights = np.zeros((*mesh.shape, 9))

        diffusion = self.compute_half_diffusion(mesh, fibers, D_al, D_ac)
        compute_weights(weights, mesh, diffusion[0, 0], diffusion[0, 1],
                        diffusion[1, 0], diffusion[1, 1])
        weights *= dt/dr**2
        weights[:, :, 4] += 1

        return weights

    def compute_half_diffusion(self, mesh, fibers, D_al, D_ac):
        D = np.zeros((2, 2, *mesh.shape))
        D[0, 0] = self.compute_diffusion_components(fibers, 0, 0, D_al, D_ac)
        D[0, 1] = self.compute_diffusion_components(fibers, 0, 1, D_al, D_ac)
        D[1, 0] = self.compute_diffusion_components(fibers, 1, 0, D_al, D_ac)
        D[1, 1] = self.compute_diffusion_components(fibers, 1, 1, D_al, D_ac)

        # D_ = np.zeros((2, *D.shape))
        # for i in range(2):
        #     #  (i-1/2, i) and (i+1/2, i)
        #     D_[0, 0, i] = 0.5 * (D[0, i] + np.roll(D[0, i], -1, axis=-2))
        #     D_[1, 0, i] = 0.5 * (D[0, i] + np.roll(D[0, i], 1, axis=-2))
        #     # (i, j+1/2) and (i, j-1/2)
        #     D_[0, 1, i] = 0.5 * (D[1, i] + np.roll(D[1, i], -1, axis=-1))
        #     D_[1, 1, i] = 0.5 * (D[1, i] + np.roll(D[1, i], 1, axis=-1))

        return D

    def compute_diffusion_components(self, fibers, ind0, ind1, D_al, D_ac):
        """
        Computes the diffusion components based on fiber orientations.

        Parameters
        ----------
        fibers : np.ndarray
            Array representing fiber orientations.
        ind0 : int
            First axis index (0 for x, 1 for y).
        ind1 : int
            Second axis index (0 for x, 1 for y).
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.

        Returns
        -------
        np.ndarray
            Array of diffusion components based on fiber orientations
        """
        return (D_ac * (ind0 == ind1)
                + (D_al - D_ac) * fibers[:, :, ind0] * fibers[:, :, ind1])
