import numpy as np
from numba import njit, prange

from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D,
    major_component,
    minor_component
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

        # q (i-1/2, j, k)
        qx0_major = major_component(d_xx[i-1, j, k], m[i-1, j, k])
        # (i-1, j, k)
        w[i, j, k, 1] += qx0_major
        # (i, j, k)
        w[i, j, k, 4] -= qx0_major

        qx0_xy_minor = minor_component(d_xy[i-1, j, k],
                                       m[i-1, j-1, k], m[i, j-1, k],
                                       m[i-1, j, k], m[i, j, k],
                                       m[i-1, j+1, k], m[i, j+1, k])
        # (i-1, j-1, k)
        w[i, j, k, 0] -= qx0_xy_minor[0]
        # (i, j-1, k)
        w[i, j, k, 3] -= qx0_xy_minor[1]
        # (i-1, j, k)
        w[i, j, k, 1] -= qx0_xy_minor[2]
        # (i, j, k)
        w[i, j, k, 4] -= qx0_xy_minor[3]
        # (i-1, j+1, k)
        w[i, j, k, 2] -= qx0_xy_minor[4]
        # (i, j+1, k)
        w[i, j, k, 5] -= qx0_xy_minor[5]

        qx0_xz_minor = minor_component(d_xz[i-1, j, k],
                                       m[i-1, j, k-1], m[i, j, k-1],
                                       m[i-1, j, k], m[i, j, k],
                                       m[i-1, j, k+1], m[i, j, k+1])
        # (i-1, j, k-1)
        w[i, j, k, 15] -= qx0_xz_minor[0]
        # (i, j, k-1)
        w[i, j, k, 11] -= qx0_xz_minor[1]
        # (i-1, j, k)
        w[i, j, k, 1] -= qx0_xz_minor[2]
        # (i, j, k)
        w[i, j, k, 4] -= qx0_xz_minor[3]
        # (i-1, j, k+1)
        w[i, j, k, 17] -= qx0_xz_minor[4]
        # (i, j, k+1)
        w[i, j, k, 12] -= qx0_xz_minor[5]

        # q (i+1/2, j, k)
        qx1_major = major_component(d_xx[i, j, k], m[i+1, j, k])
        # (i+1, j, k)
        w[i, j, k, 7] += qx1_major
        # (i, j, k)
        w[i, j, k, 4] -= qx1_major

        qx1_xy_minor = minor_component(d_xy[i, j, k],
                                       m[i+1, j-1, k], m[i, j-1, k],
                                       m[i+1, j, k], m[i, j, k],
                                       m[i+1, j+1, k], m[i, j+1, k])
        # (i+1, j-1, k)
        w[i, j, k, 6] += qx1_xy_minor[0]
        # (i, j-1, k)
        w[i, j, k, 3] += qx1_xy_minor[1]
        # (i+1, j, k)
        w[i, j, k, 7] += qx1_xy_minor[2]
        # (i, j, k)
        w[i, j, k, 4] += qx1_xy_minor[3]
        # (i+1, j+1, k)
        w[i, j, k, 8] += qx1_xy_minor[4]
        # (i, j+1, k)
        w[i, j, k, 5] += qx1_xy_minor[5]

        qx1_xz_minor = minor_component(d_xz[i, j, k],
                                       m[i+1, j, k-1], m[i, j, k-1],
                                       m[i+1, j, k], m[i, j, k],
                                       m[i+1, j, k+1], m[i, j, k+1])
        # (i+1, j, k-1)
        w[i, j, k, 16] += qx1_xz_minor[0]
        # (i, j, k-1)
        w[i, j, k, 11] += qx1_xz_minor[1]
        # (i+1, j, k)
        w[i, j, k, 7] += qx1_xz_minor[2]
        # (i, j, k)
        w[i, j, k, 4] += qx1_xz_minor[3]
        # (i+1, j, k+1)
        w[i, j, k, 18] += qx1_xz_minor[4]
        # (i, j, k+1)
        w[i, j, k, 12] += qx1_xz_minor[5]

        # q (i, j-1/2, k)
        qy0_major = major_component(d_yy[i, j-1, k], m[i, j-1, k])
        # (i, j-1, k)
        w[i, j, k, 3] += qy0_major
        # (i, j, k)
        w[i, j, k, 4] -= qy0_major

        qy0_yx_minor = minor_component(d_yx[i, j-1, k],
                                       m[i-1, j-1, k], m[i-1, j, k],
                                       m[i, j-1, k], m[i, j, k],
                                       m[i+1, j-1, k], m[i+1, j, k])
        # (i-1, j-1, k)
        w[i, j, k, 0] -= qy0_yx_minor[0]
        # (i-1, j, k)
        w[i, j, k, 1] -= qy0_yx_minor[1]
        # (i, j-1, k)
        w[i, j, k, 3] -= qy0_yx_minor[2]
        # (i, j, k)
        w[i, j, k, 4] -= qy0_yx_minor[3]
        # (i+1, j-1, k)
        w[i, j, k, 6] -= qy0_yx_minor[4]
        # (i+1, j, k)
        w[i, j, k, 7] -= qy0_yx_minor[5]

        qy0_yz_minor = minor_component(d_yz[i, j-1, k],
                                       m[i, j-1, k-1], m[i, j, k-1],
                                       m[i, j-1, k], m[i, j, k],
                                       m[i, j-1, k+1], m[i, j, k+1])
        # (i, j-1, k-1)
        w[i, j, k, 9] -= qy0_yz_minor[0]
        # (i, j, k-1)
        w[i, j, k, 11] -= qy0_yz_minor[1]
        # (i, j-1, k)
        w[i, j, k, 3] -= qy0_yz_minor[2]
        # (i, j, k)
        w[i, j, k, 4] -= qy0_yz_minor[3]
        # (i, j-1, k+1)
        w[i, j, k, 10] -= qy0_yz_minor[4]
        # (i, j, k+1)
        w[i, j, k, 12] -= qy0_yz_minor[5]

        # q (i, j+1/2, k)
        qy1_major = major_component(d_yy[i, j, k], m[i, j+1, k])
        # (i, j+1, k)
        w[i, j, k, 5] += qy1_major
        # (i, j, k)
        w[i, j, k, 4] -= qy1_major

        qy1_yx_minor = minor_component(d_yx[i, j, k],
                                       m[i-1, j+1, k], m[i-1, j, k],
                                       m[i, j+1, k], m[i, j, k],
                                       m[i+1, j+1, k], m[i+1, j, k])
        # (i-1, j+1, k)
        w[i, j, k, 2] += qy1_yx_minor[0]
        # (i-1, j, k)
        w[i, j, k, 1] += qy1_yx_minor[1]
        # (i, j+1, k)
        w[i, j, k, 5] += qy1_yx_minor[2]
        # (i, j, k)
        w[i, j, k, 4] += qy1_yx_minor[3]
        # (i+1, j+1, k)
        w[i, j, k, 8] += qy1_yx_minor[4]
        # (i+1, j, k)
        w[i, j, k, 7] += qy1_yx_minor[5]

        qy1_yz_minor = minor_component(d_yz[i, j, k],
                                       m[i, j+1, k-1], m[i, j, k-1],
                                       m[i, j+1, k], m[i, j, k],
                                       m[i, j+1, k+1], m[i, j, k+1])
        # (i, j+1, k-1)
        w[i, j, k, 13] += qy1_yz_minor[0]
        # (i, j, k-1)
        w[i, j, k, 11] += qy1_yz_minor[1]
        # (i, j+1, k)
        w[i, j, k, 5] += qy1_yz_minor[2]
        # (i, j, k)
        w[i, j, k, 4] += qy1_yz_minor[3]
        # (i, j+1, k+1)
        w[i, j, k, 14] += qy1_yz_minor[4]
        # (i, j, k+1)
        w[i, j, k, 12] += qy1_yz_minor[5]

        # q (i, j, k-1/2)
        qz0_major = major_component(d_zz[i, j, k-1], m[i, j, k-1])
        # (i, j, k-1)
        w[i, j, k, 11] += qz0_major
        # (i, j, k)
        w[i, j, k, 4] -= qz0_major

        qz0_zx_minor = minor_component(d_zx[i, j, k-1],
                                       m[i-1, j, k-1], m[i-1, j, k],
                                       m[i, j, k-1], m[i, j, k],
                                       m[i+1, j, k-1], m[i+1, j, k])
        # (i-1, j, k-1)
        w[i, j, k, 15] -= qz0_zx_minor[0]
        # (i-1, j, k)
        w[i, j, k, 1] -= qz0_zx_minor[1]
        # (i, j, k-1)
        w[i, j, k, 11] -= qz0_zx_minor[2]
        # (i, j, k)
        w[i, j, k, 4] -= qz0_zx_minor[3]
        # (i+1, j, k-1)
        w[i, j, k, 16] -= qz0_zx_minor[4]
        # (i+1, j, k)
        w[i, j, k, 7] -= qz0_zx_minor[5]

        qz0_zy_minor = minor_component(d_zy[i, j, k-1],
                                       m[i, j-1, k-1], m[i, j-1, k],
                                       m[i, j, k-1], m[i, j, k],
                                       m[i, j+1, k-1], m[i, j+1, k])
        # (i, j-1, k-1)
        w[i, j, k, 9] -= qz0_zy_minor[0]
        # (i, j-1, k)
        w[i, j, k, 3] -= qz0_zy_minor[1]
        # (i, j, k-1)
        w[i, j, k, 11] -= qz0_zy_minor[2]
        # (i, j, k)
        w[i, j, k, 4] -= qz0_zy_minor[3]
        # (i, j+1, k-1)
        w[i, j, k, 13] -= qz0_zy_minor[4]
        # (i, j+1, k)
        w[i, j, k, 5] -= qz0_zy_minor[5]

        # q (i, j, k+1/2)
        qz1_major = major_component(d_zz[i, j, k], m[i, j, k+1])
        # (i, j, k+1)
        w[i, j, k, 12] += qz1_major
        # (i, j, k)
        w[i, j, k, 4] -= qz1_major

        qz1_zx_minor = minor_component(d_zx[i, j, k],
                                       m[i-1, j, k+1], m[i-1, j, k],
                                       m[i, j, k+1], m[i, j, k],
                                       m[i+1, j, k+1], m[i+1, j, k])
        # (i-1, j, k+1)
        w[i, j, k, 17] += qz1_zx_minor[0]
        # (i-1, j, k)
        w[i, j, k, 1] += qz1_zx_minor[1]
        # (i, j, k+1)
        w[i, j, k, 12] += qz1_zx_minor[2]
        # (i, j, k)
        w[i, j, k, 4] += qz1_zx_minor[3]
        # (i+1, j, k+1)
        w[i, j, k, 18] += qz1_zx_minor[4]
        # (i+1, j, k)
        w[i, j, k, 7] += qz1_zx_minor[5]

        qz1_zy_minor = minor_component(d_zy[i, j, k],
                                       m[i, j-1, k+1], m[i, j-1, k],
                                       m[i, j, k+1], m[i, j, k],
                                       m[i, j+1, k+1], m[i, j+1, k])
        # (i, j-1, k+1)
        w[i, j, k, 10] += qz1_zy_minor[0]
        # (i, j-1, k)
        w[i, j, k, 3] += qz1_zy_minor[1]
        # (i, j, k+1)
        w[i, j, k, 12] += qz1_zy_minor[2]
        # (i, j, k)
        w[i, j, k, 4] += qz1_zy_minor[3]
        # (i, j+1, k+1)
        w[i, j, k, 14] += qz1_zy_minor[4]
        # (i, j+1, k)
        w[i, j, k, 5] += qz1_zy_minor[5]

    return w


class AsymmetricStencil3D(AsymmetricStencil2D):
    """
    A class to represent a 3D asymmetric stencil for diffusion processes.
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

        weights = compute_weights(weights, mesh, d_xx, d_xy, d_xz, d_yx, d_yy,
                                  d_yz, d_zx, d_zy, d_zz)

        weights *= dt/dr**2
        weights[:, :, :, 4] += 1

        return weights
