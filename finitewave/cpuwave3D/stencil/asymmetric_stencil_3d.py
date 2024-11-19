import numpy as np
from numba import njit, prange

from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D,
    major_component,
    compute_minor_components,
)


@njit
def compute_major_components(d_xx_0, d_xx_1, d_yy_0, d_yy_1, d_zz_0, d_zz_1,
                             m_x_0, m_x_1, m_y_0, m_y_1, m_z_0, m_z_1):
    # i-1, j, k
    w1 = major_component(d_xx_0, m_x_0)
    # i, j-1, k
    w3 = major_component(d_yy_0, m_y_0)
    # i, j+1, k
    w5 = major_component(d_yy_1, m_y_1)
    # i+1, j, k
    w7 = major_component(d_xx_1, m_x_1)
    # i, j, k-1
    w11 = major_component(d_zz_0, m_z_0)
    # i, j, k+1
    w12 = major_component(d_zz_1, m_z_1)
    return w1, w3, w5, w7, w11, w12


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

        w_major = compute_major_components(d_xx[i-1, j, k], d_xx[i, j, k],
                                           d_yy[i, j-1, k], d_yy[i, j, k],
                                           d_zz[i, j, k-1], d_zz[i, j, k],
                                           m[i-1, j, k], m[i+1, j, k],
                                           m[i, j-1, k], m[i, j+1, k],
                                           m[i, j, k-1], m[i, j, k+1])

        w_minor_xy = compute_minor_components(d_xy[i-1, j, k], d_xy[i, j, k],
                                              d_yx[i, j-1, k], d_yx[i, j, k],
                                              m[i-1, j-1, k], m[i-1, j, k],
                                              m[i-1, j+1, k], m[i, j-1, k],
                                              m[i, j+1, k], m[i+1, j-1, k],
                                              m[i+1, j, k], m[i+1, j+1, k])

        w_minor_yz = compute_minor_components(d_yz[i, j-1, k], d_yz[i, j, k],
                                              d_zy[i, j, k-1], d_zy[i, j, k],
                                              m[i, j-1, k-1], m[i, j-1, k],
                                              m[i, j-1, k+1], m[i, j, k-1],
                                              m[i, j, k+1], m[i, j+1, k-1],
                                              m[i, j+1, k], m[i, j+1, k+1])

        w_minor_zx = compute_minor_components(d_zx[i, j, k-1], d_zx[i, j, k],
                                              d_xz[i-1, j, k], d_xz[i, j, k],
                                              m[i-1, j, k-1], m[i, j, k-1],
                                              m[i+1, j, k-1], m[i-1, j, k],
                                              m[i+1, j, k], m[i-1, j, k+1],
                                              m[i, j, k+1], m[i+1, j, k+1])
        # Add major components
        # i-1, j, k
        w[i, j, k, 1] = w_major[0]
        # i, j-1, k
        w[i, j, k, 3] = w_major[1]
        # i, j, k
        w[i, j, k, 4] = - sum(w_major)
        # i, j+1, k
        w[i, j, k, 5] = w_major[2]
        # i+1, j, k
        w[i, j, k, 7] = w_major[3]
        # i, j, k-1
        w[i, j, k, 11] = w_major[4]
        # i, j, k+1
        w[i, j, k, 12] = w_major[5]
        
        # Add minor xy components
        # i-1, j-1, k
        w[i, j, k, 0] += w_minor_xy[0]
        # i-1, j, k
        w[i, j, k, 1] += w_minor_xy[1]
        # i-1, j+1, k
        w[i, j, k, 2] += w_minor_xy[2]
        # i, j-1, k
        w[i, j, k, 3] += w_minor_xy[3]
        # i, j+1, k
        w[i, j, k, 5] += w_minor_xy[4]
        # i+1, j-1, k
        w[i, j, k, 6] += w_minor_xy[5]
        # i+1, j, k
        w[i, j, k, 7] += w_minor_xy[6]
        # i+1, j+1, k
        w[i, j, k, 8] += w_minor_xy[7]
        
        # Add minor yz components
        # i, j-1, k-1
        w[i, j, k, 9] += w_minor_yz[0]
        # i, j-1, k
        w[i, j, k, 3] += w_minor_yz[1]
        # i, j-1, k+1
        w[i, j, k, 10] += w_minor_yz[2]
        # i, j, k-1
        w[i, j, k, 11] += w_minor_yz[3]
        # i, j, k+1
        w[i, j, k, 12] += w_minor_yz[4]
        # i, j+1, k-1
        w[i, j, k, 13] += w_minor_yz[5]
        # i, j+1, k
        w[i, j, k, 5] += w_minor_yz[6]
        # i, j+1, k+1
        w[i, j, k, 14] += w_minor_yz[7]

        # Add minor zx components
        # i-1, j, k-1
        w[i, j, k, 15] += w_minor_zx[0]
        # i, j, k-1
        w[i, j, k, 11] += w_minor_zx[1]
        # i+1, j, k-1
        w[i, j, k, 16] += w_minor_zx[2]
        # i-1, j, k
        w[i, j, k, 1] += w_minor_zx[3]
        # i+1, j, k
        w[i, j, k, 7] += w_minor_zx[4]
        # i-1, j, k+1
        w[i, j, k, 17] += w_minor_zx[5]
        # i, j, k+1
        w[i, j, k, 12] += w_minor_zx[6]
        # i+1, j, k+1
        w[i, j, k, 18] += w_minor_zx[7]

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
