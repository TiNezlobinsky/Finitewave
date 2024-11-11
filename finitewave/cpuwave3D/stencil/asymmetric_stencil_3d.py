import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    compute_local_weights,
    AsymmetricStencil2D
)


@njit
def compute_weights(w, m, d_xx, d_xy, d_xz, d_yx, d_yy, d_yz, d_zx, d_zy,
                    d_zz):
    """
    Computes the weights for diffusion on a 3D mesh using an asymmetric
    stencil.

    Parameters
    ----------
    w : np.ndarray
        4D array of weights for diffusion, with the shape of (*mesh.shape, 19).
    m : np.ndarray
        3D array representing the mesh grid of the tissue.
        Non-tissue areas are set to 0.
    d_xx : np.ndarray
        3D array of half-step diffusion x-components in the x-direction.
    d_xy : np.ndarray
        3D array of half-step diffusion y-components in the x-direction.
    d_xz : np.ndarray
        3D array of half-step diffusion z-components in the x-direction.
    d_yx : np.ndarray
        3D array of half-step diffusion x-components in the y-direction.
    d_yy : np.ndarray
        3D array of half-step diffusion y-components in the y-direction.
    d_yz : np.ndarray
        3D array of half-step diffusion z-components in the y-direction.
    d_zx : np.ndarray
        3D array of half-step diffusion x-components in the z-direction.
    d_zy : np.ndarray
        3D array of half-step diffusion y-components in the z-direction.
    d_zz : np.ndarray
        3D array of half-step diffusion z-components in the z-direction.

    Returns
    -------
    np.ndarray
        4D array of weights for diffusion, with the shape of (*mesh.shape, 9).
    """
    n_i = m.shape[0]
    n_j = m.shape[1]
    n_k = m.shape[2]
    for ii in prange(n_i*n_j*n_k):
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        if m[i, j, k] != 1:
            continue

        res_xy = compute_local_weights(d_xx[i-1, j, k], d_xx[i, j, k],
                                       d_xy[i-1, j, k], d_xy[i, j, k],
                                       d_yx[i, j-1, k], d_yx[i, j, k],
                                       d_yy[i, j-1, k], d_yy[i, j, k],
                                       m[i-1, j-1, k], m[i-1, j, k],
                                       m[i-1, j+1, k], m[i, j-1, k],
                                       m[i, j, k], m[i, j+1, k],
                                       m[i+1, j-1, k], m[i+1, j, k],
                                       m[i+1, j+1, k])

        res_yz = compute_local_weights(d_yy[i, j-1, k], d_yy[i, j, k],
                                       d_yz[i, j-1, k], d_yz[i, j, k],
                                       d_zy[i, j, k-1], d_zy[i, j, k],
                                       d_zz[i, j, k-1], d_zz[i, j, k],
                                       m[i, j-1, k-1], m[i, j-1, k],
                                       m[i, j-1, k+1], m[i, j, k-1],
                                       m[i, j, k], m[i, j, k+1],
                                       m[i, j+1, k-1], m[i, j+1, k],
                                       m[i, j+1, k+1])

        res_zx = compute_local_weights(d_zz[i, j, k-1], d_zz[i, j, k],
                                       d_zx[i, j, k-1], d_zx[i, j, k],
                                       d_xz[i-1, j, k], d_xz[i, j, k],
                                       d_xx[i-1, j, k], d_xx[i, j, k],
                                       m[i-1, j, k-1], m[i, j, k-1],
                                       m[i+1, j, k-1], m[i-1, j, k],
                                       m[i, j, k], m[i+1, j, k],
                                       m[i-1, j, k+1], m[i, j, k+1],
                                       m[i+1, j, k+1])
        # i-1, j-1, k
        w[i, j, k, 0] = res_xy[0]
        # i-1, j, k
        w[i, j, k, 1] = res_xy[1] + res_zx[3]
        # i-1, j+1, k
        w[i, j, k, 2] = res_xy[2]
        # i, j-1, k
        w[i, j, k, 3] = res_xy[3] + res_yz[1]
        # i, j, k
        w[i, j, k, 4] = res_xy[4] + res_yz[4] + res_zx[4]
        # i, j+1, k
        w[i, j, k, 5] = res_xy[5] + res_yz[6]
        # i+1, j-1, k
        w[i, j, k, 6] = res_xy[6]
        # i+1, j, k
        w[i, j, k, 7] = res_xy[7] + res_zx[5]
        # i+1, j+1, k
        w[i, j, k, 8] = res_xy[8]

        # i, j-1, k-1
        w[i, j, k, 9] = res_yz[0]
        # i, j-1, k+1
        w[i, j, k, 10] = res_yz[2]
        # i, j, k-1
        w[i, j, k, 11] = res_yz[3] + res_zx[1]
        # i, j, k+1
        w[i, j, k, 12] = res_yz[4] + res_zx[7]
        # i, j+1, k-1
        w[i, j, k, 13] = res_yz[5]
        # i, j+1, k+1
        w[i, j, k, 14] = res_yz[7]

        # i-1, j, k-1
        w[i, j, k, 15] = res_zx[0]
        # i+1, j, k-1
        w[i, j, k, 16] = res_zx[2]
        # i-1, j, k+1
        w[i, j, k, 17] = res_zx[6]
        # i+1, j, k+1
        w[i, j, k, 18] = res_zx[8]

    return w


class AsymmetricStencil3D(AsymmetricStencil2D):
    """
    A class to represent a 3D asymmetric stencil for diffusion processes.

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
        Initializes the AsymmetricStencil3D with default settings.
        """
        super().__init__()

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        """
        Computes the weights for diffusion on a 3D mesh using an asymmetric stencil.

        Parameters
        ----------
        mesh : np.ndarray
            3D array representing the mesh grid of the tissue. Non-tissue areas are set to 0.
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
            4D array of weights for diffusion, with the shape of (mesh.shape[0], mesh.shape[1], 9).

        Notes
        -----
        The method assumes asymmetric diffusion where different coefficients are used for different directions.
        The weights are computed for eight surrounding directions and the central weight, based on the asymmetric stencil.
        Heterogeneity in the diffusion coefficients is handled by adjusting the weights based on fiber orientations.
        """
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        weights = np.zeros((*mesh.shape, 19), dtype='float32')

        d_xx, d_xy, d_xz = self.compute_half_step_diffusion(mesh, conductivity,
                                                            fibers, D_al, D_ac,
                                                            0, num_axes=3)
        d_yx, d_yy, d_yz = self.compute_half_step_diffusion(mesh, conductivity,
                                                            fibers, D_al, D_ac,
                                                            1, num_axes=3)
        d_zx, d_zy, d_zz = self.compute_half_step_diffusion(mesh, conductivity,
                                                            fibers, D_al, D_ac,
                                                            2, num_axes=3)

        weights = compute_weights(weights, mesh, d_xx, d_xy, d_xz, d_yy, d_yx,
                                  d_yz, d_zz, d_zx, d_zy)

        weights *= dt/dr**2
        weights[:, :, :, 4] += 1

        return weights.astype('float32')
