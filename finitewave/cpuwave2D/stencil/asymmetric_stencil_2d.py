import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


@njit
def coeffs(m0, m1, m2, m3):
    """
    Computes the coefficients for secondary partial derivatives.

    For example for corner point ``i-1, j-1`` which is used in the calculation
    of the ``du/dy`` at the point ``o = (i-1/2, j)``:

    .. code-block:: text
        m1 ----- m3 --- x
        |         |     |
        x -- o -- x --- x
        |         |     |
        m0 ------ m2 -- x

    Coefficient ``m0 == 0`` if:
    - ``m0 == 0``
    - ``m1 == 0`` and ``m3 == 0``

    Coefficient ``m0 == 1`` if:
    - ``m0 == 1`` and ``m2 == 1``

    Coefficient ``m0 == 2`` if:
    - ``m0 == 1`` and ``m2 == 0``

    Parameters
    ----------
    m0 : int
        Target mesh point value.
    m1 : int
        Mesh point opposite to m0.
    m2 : int
        Mesh point adjacent to m0.
    m3 : int
        Mesh point adjacent to m1.

    Returns
    -------
    int
        Coefficient for the secondary partial derivatives.
    """
    return m0 * (m0 + (m2 == 0)) * ((m1 + m3) >= 1)


@njit
def compute_weights(w, m, d_xx, d_xy, d_yx, d_yy):
    """
    Computes the weights for diffusion on a 2D mesh based on the asymmetric
    stencil.

    Parameters
    ----------
    w : np.ndarray
        3D array to store the weights for diffusion. Shape is (*mesh.shape, 9).
    m : np.ndarray
        2D array representing the mesh grid of the tissue. Non-tissue areas
        are set to 0.
    d_xx : np.ndarray
        Diffusion x component for x direction.
    d_xy : np.ndarray
        Diffusion y component for x direction.
    d_yx : np.ndarray
        Diffusion x component for y direction.
    d_yy : np.ndarray
        Diffusion y component for y direction.

    Returns
    -------
    np.ndarray
        3D array of weights for diffusion, with the shape of (*mesh.shape, 9).

    Notes
    -----
    The method assumes weights being used in the following order:
        ``w[0] : i-1, j-1``,
        ``w[1] : i-1, j``,
        ``w[2] : i-1, j+1``,
        ``w[3] : i, j-1``,
        ``w[4] : i, j``,
        ``w[5] : i, j+1``,
        ``w[6] : i+1, j-1``,
        ``w[7] : i+1, j``,
        ``w[8] : i+1, j+1``.
    """
    n_i = m.shape[0]
    n_j = m.shape[1]
    for ii in prange(n_i * n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if m[i, j] != 1:
            continue

        # i-1, j-1
        w[i, j, 0] = 0.25 * (d_xy[i-1, j] * m[i-1, j] * coeffs(m[i-1, j-1],
                                                               m[i-1, j+1],
                                                               m[i, j-1],
                                                               m[i, j+1]) +
                             d_yx[i, j-1] * m[i, j-1] * coeffs(m[i-1, j-1],
                                                               m[i+1, j-1],
                                                               m[i-1, j],
                                                               m[i+1, j]))
        # i-1, j
        w[i, j, 1] = (d_xx[i-1, j] * m[i-1, j] +
                      0.25 * (d_yx[i, j-1] * m[i, j-1] * coeffs(m[i-1, j],
                                                                m[i+1, j],
                                                                m[i-1, j-1],
                                                                m[i+1, j-1]) -
                              d_yx[i, j] * m[i, j+1] * coeffs(m[i-1, j],
                                                              m[i+1, j],
                                                              m[i-1, j+1],
                                                              m[i+1, j+1])))
        # i-1, j+1
        w[i, j, 2] = -0.25 * (d_xy[i-1, j] * m[i-1, j] * coeffs(m[i-1, j+1],
                                                                m[i-1, j-1],
                                                                m[i, j+1],
                                                                m[i, j-1]) +
                              d_yx[i, j] * m[i, j+1] * coeffs(m[i-1, j+1],
                                                              m[i+1, j+1],
                                                              m[i-1, j],
                                                              m[i+1, j]))
        # i, j-1
        w[i, j, 3] = (d_yy[i, j-1] * m[i, j-1] +
                      0.25 * (d_xy[i-1, j] * m[i-1, j] * coeffs(m[i, j-1],
                                                                m[i, j+1],
                                                                m[i-1, j-1],
                                                                m[i-1, j+1]) -
                              d_xy[i, j] * m[i+1, j] * coeffs(m[i, j-1],
                                                              m[i, j+1],
                                                              m[i+1, j-1],
                                                              m[i+1, j+1])))
        # i, j
        w[i, j, 4] = - (d_xx[i-1, j] * m[i-1, j] + d_xx[i, j] * m[i+1, j] +
                        d_yy[i, j-1] * m[i, j-1] + d_yy[i, j] * m[i, j+1])
        # i, j+1
        w[i, j, 5] = (d_yy[i, j] * m[i, j+1] +
                      0.25 * (-d_xy[i-1, j] * m[i-1, j] * coeffs(m[i, j+1],
                                                                 m[i, j-1],
                                                                 m[i-1, j+1],
                                                                 m[i-1, j-1]) +
                              d_xy[i, j] * m[i+1, j] * coeffs(m[i, j+1],
                                                              m[i, j-1],
                                                              m[i+1, j+1],
                                                              m[i+1, j-1])))
        # i+1, j-1
        w[i, j, 6] = -0.25 * (d_xy[i, j] * m[i+1, j] * coeffs(m[i+1, j-1],
                                                              m[i+1, j+1],
                                                              m[i, j-1],
                                                              m[i, j+1]) +
                              d_yx[i, j-1] * m[i, j-1] * coeffs(m[i+1, j-1],
                                                                m[i-1, j-1],
                                                                m[i+1, j],
                                                                m[i-1, j]))
        # i+1, j
        w[i, j, 7] = (d_xx[i, j] * m[i+1, j] +
                      0.25 * (-d_yx[i, j-1] * m[i, j-1] * coeffs(m[i+1, j],
                                                                 m[i-1, j],
                                                                 m[i+1, j-1],
                                                                 m[i-1, j-1]) +
                              d_yx[i, j] * m[i, j+1] * coeffs(m[i+1, j],
                                                              m[i-1, j],
                                                              m[i+1, j+1],
                                                              m[i-1, j+1])))
        # i+1, j+1
        w[i, j, 8] = 0.25 * (d_xy[i, j] * m[i+1, j] * coeffs(m[i+1, j+1],
                                                             m[i+1, j-1],
                                                             m[i, j+1],
                                                             m[i, j-1]) +
                             d_yx[i, j] * m[i, j+1] * coeffs(m[i+1, j+1],
                                                             m[i-1, j+1],
                                                             m[i+1, j],
                                                             m[i-1, j]))


class AsymmetricStencil2D(Stencil):
    """
    A class to represent a 2D asymmetric stencil for diffusion processes.
    The asymmetric stencil is used to handle anisotropic diffusion in the
    tissue.
    """

    def __init__(self):
        super().__init__()

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        """
        Computes the weights for diffusion on a 2D mesh using an asymmetric
        stencil.

        Parameters
        ----------
        mesh : np.ndarray
            2D array representing the mesh grid of the tissue.
        conductivity : float
            Conductivity of the tissue, which scales the diffusion coefficient.
        fibers : np.ndarray
            Array representing fiber orientations. Used to compute directional
            diffusion coefficients.
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

        diffusion = self.compute_half_step_diffusion(mesh, conductivity,
                                                     fibers, D_al, D_ac)
        diffusion *= conductivity
        compute_weights(weights, mesh, diffusion[0, 0], diffusion[0, 1],
                        diffusion[1, 0], diffusion[1, 1])
        weights *= dt/dr**2
        weights[:, :, 4] += 1

        return weights

    def compute_half_step_diffusion(self, mesh, conductivity, fibers, D_al,
                                    D_ac):
        """
        Computes the diffusion components for half-steps based on fiber
        orientations.

        Parameters
        ----------
        mesh : np.ndarray
            Array representing the mesh grid of the tissue.
        conductivity : np.ndarray
            Array representing the conductivity of the tissue.
        fibers : np.ndarray
            Array representing fiber orientations with shape
            ``(*mesh.shape, 2)``.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.

        Returns
        -------
        np.ndarray
            4D array of diffusion components for half-steps based on fiber
            orientations. Shape is (2, 2, *mesh.shape). The index ``(i, j)``
            corresponds to ``(i+1/2, j)`` or ``(i, j+1/2)`` half-steps
            depending on the diffusion component. Thus, ``(i-1, j)`` and
            ``(i, j-1)`` correspond to the ``(i-1/2, j)`` and ``(i, j-1/2)``
            half-steps, respectively.
        """
        D = np.zeros((2, 2, *mesh.shape))

        D[0, 0] = self.compute_diffusion_components(fibers, 0, 0, D_al, D_ac)
        D[0, 1] = self.compute_diffusion_components(fibers, 0, 1, D_al, D_ac)
        D[1, 0] = self.compute_diffusion_components(fibers, 1, 0, D_al, D_ac)
        D[1, 1] = self.compute_diffusion_components(fibers, 1, 1, D_al, D_ac)

        if conductivity is not None:
            D *= conductivity

        # (i-1/2, j) and (i+1/2, j)
        D[0, 0] = 0.5 * (D[0, 0] + np.roll(D[0, 0], -1, axis=-2))
        D[0, 1] = 0.5 * (D[0, 1] + np.roll(D[0, 1], -1, axis=-2))
        # (i, j-1/2) and (i, j+1/2)
        D[1, 0] = 0.5 * (D[1, 0] + np.roll(D[1, 0], -1, axis=-1))
        D[1, 1] = 0.5 * (D[1, 1] + np.roll(D[1, 1], -1, axis=-1))

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
