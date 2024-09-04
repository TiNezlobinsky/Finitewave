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
def compute_weights(w, m, d_x, d_xy, d_y, d_yx):
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
        fibers[np.where(mesh != 1)] = 0
        weights = np.zeros((*mesh.shape, 9))

        def axis_fibers(fibers, ind):
            """
            Computes fiber directions for a given axis.

            Parameters
            ----------
            fibers : np.ndarray
                Array representing fiber orientations.
            ind : int
                Axis index (0 for x, 1 for y).

            Returns
            -------
            np.ndarray
                Normalized fiber directions along the specified axis.
            """
            fibr = fibers + np.roll(fibers, 1, axis=ind)
            norm = np.linalg.norm(fibr, axis=2)
            np.divide(fibr, norm[:, :, np.newaxis], out=fibr,
                      where=norm[:, :, np.newaxis] != 0)
            return fibr

        def major_diffuse(fibers, ind):
            """
            Computes the major diffusion term based on fiber orientations.

            Parameters
            ----------
            fibers : np.ndarray
                Array representing fiber orientations.
            ind : int
                Axis index (0 for x, 1 for y).

            Returns
            -------
            np.ndarray
                Array of major diffusion coefficients.
            """
            return ((D_ac + (D_al - D_ac) * fibers[:, :, ind]**2) *
                    conductivity)

        def minor_diffuse(fibers, ind1, ind2):
            """
            Computes the minor diffusion term based on fiber orientations.

            Parameters
            ----------
            fibers : np.ndarray
                Array representing fiber orientations.
            ind1 : int
                First axis index (0 for x, 1 for y).
            ind2 : int
                Second axis index (0 for x, 1 for y).

            Returns
            -------
            np.ndarray
                Array of minor diffusion coefficients.
            """
            return (0.5 * (D_al - D_ac) * fibers[:, :, ind1] *
                    fibers_x[:, :, ind2] * conductivity)

        fibers_x = axis_fibers(fibers, 0)
        fibers_y = axis_fibers(fibers, 1)

        diffuse_x = major_diffuse(fibers_x, 0)
        diffuse_xy = minor_diffuse(fibers_x, 0, 1)

        diffuse_y = major_diffuse(fibers_y, 1)
        diffuse_yx = minor_diffuse(fibers_y, 1, 0)

        compute_weights(weights, mesh, diffuse_x, diffuse_xy, diffuse_y,
                        diffuse_yx)
        weights *= dt/dr**2
        weights[:, :, 4] += 1

        return weights
