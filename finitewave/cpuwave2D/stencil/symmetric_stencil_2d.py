import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


@njit
def compute_component(m0, m1, m2, m3, d_xx, d_xy, d_yx, d_yy, qx, qy, ux, uy):
    m = m0 * m1 * m2 * m3
    w = (qx * (d_xx * ux + d_xy * uy) + qy * (d_yx * ux + d_yy * uy)) * m
    return 0.25 * w

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
        # (i-1, j-1)
        w[i, j, 0] = compute_component(m[i-1, j-1], m[i-1, j], m[i, j-1], m[i, j],
                                       d_xx[i-1, j-1], d_xy[i-1, j-1], d_yx[i-1, j-1], d_yy[i-1, j-1],
                                       -1, -1, -1, -1)
        # (i-1, j)
        w[i, j, 1] = (compute_component(m[i-1, j-1], m[i-1, j], m[i, j-1], m[i, j],
                                        d_xx[i-1, j-1], d_xy[i-1, j-1], d_yx[i-1, j-1], d_yy[i-1, j-1],
                                        -1, -1, -1, 1) +
                      compute_component(m[i-1, j], m[i-1, j+1], m[i, j], m[i, j+1],
                                        d_xx[i-1, j], d_xy[i-1, j], d_yx[i-1, j], d_yy[i-1, j],
                                        -1, 1, -1, -1))
        # (i-1, j+1)
        w[i, j, 2] = compute_component(m[i-1, j], m[i-1, j+1], m[i, j], m[i, j+1],
                                        d_xx[i-1, j], d_xy[i-1, j], d_yx[i-1, j], d_yy[i-1, j],
                                        -1, 1, -1, 1)
        # (i, j-1)
        w[i, j, 3] = (compute_component(m[i-1, j-1], m[i-1, j], m[i, j-1], m[i, j],
                                        d_xx[i-1, j-1], d_xy[i-1, j-1], d_yx[i-1, j-1], d_yy[i-1, j-1],
                                        -1, -1, 1, -1) +
                        compute_component(m[i, j-1], m[i, j], m[i+1, j-1], m[i+1, j],
                                        d_xx[i, j-1], d_xy[i, j-1], d_yx[i, j-1], d_yy[i, j-1],
                                        1, -1, -1, -1))
        # (i, j)
        w[i, j, 4] = (compute_component(m[i-1, j-1], m[i-1, j], m[i, j-1], m[i, j],
                                        d_xx[i-1, j-1], d_xy[i-1, j-1], d_yx[i-1, j-1], d_yy[i-1, j-1],
                                        -1, -1, 1, 1) +
                      compute_component(m[i-1, j], m[i-1, j+1], m[i, j], m[i, j+1],
                                        d_xx[i-1, j], d_xy[i-1, j], d_yx[i-1, j], d_yy[i-1, j],
                                        -1, 1, 1, -1) +
                      compute_component(m[i, j-1], m[i, j], m[i+1, j-1], m[i+1, j],
                                        d_xx[i, j-1], d_xy[i, j-1], d_yx[i, j-1], d_yy[i, j-1],
                                        1, -1, -1, 1) +
                      compute_component(m[i, j], m[i, j+1], m[i+1, j], m[i+1, j+1],
                                        d_xx[i, j], d_xy[i, j], d_yx[i, j], d_yy[i, j],
                                        1, 1, -1, -1))
        # (i, j+1)
        w[i, j, 5] = (compute_component(m[i, j], m[i, j+1], m[i+1, j], m[i+1, j+1],
                                        d_xx[i, j], d_xy[i, j], d_yx[i, j], d_yy[i, j],
                                        1, 1, -1, 1) +
                      compute_component(m[i-1, j], m[i-1, j+1], m[i, j], m[i, j+1],
                                        d_xx[i-1, j], d_xy[i-1, j], d_yx[i-1, j], d_yy[i-1, j],
                                        -1, 1, 1, 1))
        # (i+1, j-1)
        w[i, j, 6] = compute_component(m[i, j-1], m[i, j], m[i+1, j-1], m[i+1, j],
                                        d_xx[i, j-1], d_xy[i, j-1], d_yx[i, j-1], d_yy[i, j-1],
                                        1, -1, 1, -1)
        # (i+1, j)
        w[i, j, 7] = (compute_component(m[i, j], m[i, j+1], m[i+1, j], m[i+1, j+1],
                                        d_xx[i, j], d_xy[i, j], d_yx[i, j], d_yy[i, j],
                                        1, 1, 1, -1) +
                      compute_component(m[i, j-1], m[i, j], m[i+1, j-1], m[i+1, j],
                                        d_xx[i, j-1], d_xy[i, j-1], d_yx[i, j-1], d_yy[i, j-1],
                                        1, -1, 1, 1))
        # (i+1, j+1)
        w[i, j, 8] = compute_component(m[i, j], m[i, j+1], m[i+1, j], m[i+1, j+1],
                                        d_xx[i, j], d_xy[i, j], d_yx[i, j], d_yy[i, j],
                                        1, 1, 1, 1)

    return w


class SymmetricStencil2D(Stencil):
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

        D = self.compute_half_step_diffusion(mesh, conductivity, fibers, D_al,
                                             D_ac)
        compute_weights(weights, mesh, D[0], D[1], D[2], D[3])
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
            ``(2, *mesh.shape)``.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.
        axis : int
            Axis index (0 for x, 1 for y).
        num_axes : int
            Number of axes.

        Returns
        -------
        np.ndarray
            Array of diffusion components for half-steps along the specified
            axis.

        Notes
        -----
        The index ``i`` in the returned array corresponds to ``i+1/2`` and
        ``i-1`` corresponds to ``i-1/2``.
        """
        if conductivity is None:
            conductivity = 1

        D = np.zeros((4, *mesh.shape))
        for i in range(4):
            ind0 = i // 2
            ind1 = i % 2
            D[i] = self.compute_diffusion_components(fibers, ind0, ind1, D_al,
                                                     D_ac)
            D[i] *= conductivity
            D[i] = 0.25 * (D[i] +
                           np.roll(D[i], -1, axis=0) +
                           np.roll(D[i], -1, axis=1) +
                           np.roll(np.roll(D[i], -1, axis=0), -1, axis=1))

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
        return (D_ac * (ind0 == ind1) +
                (D_al - D_ac) * fibers[ind0] * fibers[ind1])
