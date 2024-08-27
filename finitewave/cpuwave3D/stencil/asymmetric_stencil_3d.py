import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


@njit
def coeffs(d, m, m0, m1, m2, m3):
    return 0.5 * d * m * m0 * m1 / (1 + m0 * m1 * m2 * m3)


@njit
def compute_weights(w, m, d_x, d_xy, d_xz, d_y, d_yx, d_yz, d_z, d_zx, d_zy):
    n_i = m.shape[0]
    n_j = m.shape[1]
    n_k = m.shape[2]
    for ii in prange(n_i*n_j*n_k):
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k
        if m[i, j, k] != 1:
            continue

        w[i, j, k, 0] = (coeffs(d_xy[i-1, j, k], m[i-1, j, k], m[i-1, j-1, k],
                                m[i-1, j+1, k], m[i, j-1, k], m[i, j+1, k]) +
                         coeffs(d_yx[i, j-1, k], m[i, j-1, k], m[i-1, j-1, k],
                                m[i+1, j-1, k], m[i-1, j, k], m[i+1, j, k]))

        w[i, j, k, 1] = (coeffs(d_xz[i-1, j, k], m[i-1, j, k], m[i-1, j, k-1],
                                m[i-1, j, k+1], m[i, j, k-1], m[i, j, k+1]) +
                         coeffs(d_zx[i, j, k-1], m[i, j, k-1], m[i-1, j, k-1],
                                m[i+1, j, k-1], m[i-1, j, k], m[i+1, j, k]))

        w[i, j, k, 2] = (d_x[i-1, j, k] * m[i-1, j, k] +
                         coeffs(d_yx[i, j-1, k], m[i, j-1, k], m[i-1, j, k],
                                m[i+1, j, k], m[i-1, j-1, k], m[i+1, j-1, k]) +
                         coeffs(-d_yx[i, j, k], m[i, j+1, k], m[i-1, j, k],
                                m[i+1, j, k], m[i-1, j+1, k], m[i+1, j+1, k]) +
                         coeffs(d_zx[i, j, k-1], m[i, j, k-1], m[i-1, j, k],
                                m[i+1, j, k], m[i-1, j, k-1], m[i+1, j, k-1]) +
                         coeffs(-d_zx[i, j, k], m[i, j, k+1], m[i-1, j, k],
                                m[i+1, j, k], m[i-1, j, k+1], m[i+1, j, k+1]))

        w[i, j, k, 3] = (coeffs(-d_xz[i-1, j, k], m[i-1, j, k], m[i-1, j, k-1],
                                m[i-1, j, k+1], m[i, j, k-1], m[i, j, k+1]) +
                         coeffs(-d_zx[i, j, k], m[i, j, k+1], m[i-1, j, k+1],
                                m[i+1, j, k+1], m[i-1, j, k], m[i+1, j, k]))

        w[i, j, k, 4] = (coeffs(-d_xy[i-1, j, k], m[i-1, j, k], m[i-1, j-1, k],
                                m[i-1, j+1, k], m[i, j-1, k], m[i, j+1, k]) +
                         coeffs(-d_yx[i, j, k], m[i, j+1, k], m[i-1, j+1, k],
                                m[i+1, j+1, k], m[i-1, j, k], m[i+1, j, k]))

        w[i, j, k, 5] = (coeffs(d_yz[i, j-1, k], m[i, j-1, k], m[i, j-1, k-1],
                                m[i, j-1, k+1], m[i, j, k-1], m[i, j, k+1]) +
                         coeffs(d_zy[i, j, k-1], m[i, j, k-1], m[i, j-1, k-1],
                                m[i, j+1, k-1], m[i, j-1, k], m[i, j+1, k]))

        w[i, j, k, 6] = (d_y[i, j-1, k] * m[i, j-1, k] +
                         coeffs(d_xy[i-1, j, k], m[i-1, j, k], m[i, j-1, k],
                                m[i, j+1, k], m[i-1, j-1, k], m[i-1, j+1, k]) +
                         coeffs(-d_xy[i, j, k], m[i+1, j, k], m[i, j-1, k],
                                m[i, j+1, k], m[i+1, j-1, k], m[i+1, j+1, k]) +
                         coeffs(d_zy[i, j, k-1], m[i, j, k-1], m[i, j-1, k],
                                m[i, j+1, k], m[i, j-1, k-1], m[i, j+1, k-1]) +
                         coeffs(-d_zy[i, j, k], m[i, j, k+1], m[i, j-1, k],
                                m[i, j+1, k], m[i, j-1, k+1], m[i, j+1, k+1]))

        w[i, j, k, 7] = (coeffs(-d_yz[i, j-1, k], m[i, j-1, k], m[i, j-1, k-1],
                                m[i, j-1, k+1], m[i, j, k-1], m[i, j, k+1]) +
                         coeffs(-d_zy[i, j, k], m[i, j, k+1], m[i, j-1, k+1],
                                m[i, j+1, k+1], m[i, j-1, k], m[i, j+1, k]))

        w[i, j, k, 8] = (d_z[i, j, k-1] * m[i, j, k-1] +
                         coeffs(d_yz[i, j-1, k], m[i, j-1, k], m[i, j, k-1],
                                m[i, j, k+1], m[i, j-1, k-1], m[i, j-1, k+1]) +
                         coeffs(-d_yz[i, j, k], m[i, j+1, k], m[i, j, k-1],
                                m[i, j, k+1], m[i, j+1, k-1], m[i, j+1, k+1]) +
                         coeffs(d_xz[i-1, j, k], m[i-1, j, k], m[i, j, k-1],
                                m[i, j, k+1], m[i-1, j, k-1], m[i-1, j, k+1]) +
                         coeffs(-d_xz[i, j, k], m[i+1, j, k], m[i, j, k-1],
                                m[i, j, k+1], m[i+1, j, k-1], m[i+1, j, k+1]))

        w[i, j, k, 9] = - (m[i-1, j, k] * d_x[i-1, j, k] +
                           m[i+1, j, k] * d_x[i, j, k] +
                           m[i, j-1, k] * d_y[i, j-1, k] +
                           m[i, j+1, k] * d_y[i, j, k] +
                           m[i, j, k-1] * d_z[i, j, k-1] +
                           m[i, j, k+1] * d_z[i, j, k])

        w[i, j, k, 10] = (d_z[i, j, k] * m[i, j, k+1] +
                          coeffs(-d_xz[i-1, j, k], m[i-1, j, k], m[i, j, k-1],
                                 m[i, j, k+1], m[i-1, j, k-1], m[i-1, j, k+1]) +
                          coeffs(d_xz[i, j, k], m[i+1, j, k], m[i, j, k-1],
                                 m[i, j, k+1], m[i+1, j, k-1], m[i+1, j, k+1]) +
                          coeffs(-d_yz[i, j-1, k], m[i, j-1, k], m[i, j, k-1],
                                 m[i, j, k+1], m[i, j-1, k-1], m[i, j-1, k+1]) +
                          coeffs(d_yz[i, j, k], m[i, j+1, k], m[i, j, k-1],
                                 m[i, j, k+1], m[i, j+1, k-1], m[i, j+1, k+1]))

        w[i, j, k, 11] = (coeffs(-d_yz[i, j, k], m[i, j+1, k], m[i, j+1, k-1],
                                 m[i, j+1, k+1], m[i, j, k-1], m[i, j, k+1]) +
                          coeffs(-d_zy[i, j, k-1], m[i, j, k-1], m[i, j-1, k-1],
                                 m[i, j+1, k-1], m[i, j-1, k], m[i, j+1, k]))

        w[i, j, k, 12] = (d_y[i, j, k] * m[i, j+1, k] +
                          coeffs(-d_xy[i-1, j, k], m[i-1, j, k], m[i, j-1, k],
                                 m[i, j+1, k], m[i-1, j-1, k], m[i-1, j+1, k]) +
                          coeffs(d_xy[i, j, k], m[i+1, j, k], m[i, j-1, k],
                                 m[i, j+1, k], m[i+1, j-1, k], m[i+1, j+1, k]) +
                          coeffs(-d_zy[i, j, k-1], m[i, j, k-1], m[i, j-1, k],
                                 m[i, j+1, k], m[i, j-1, k-1], m[i, j+1, k-1]) +
                          coeffs(d_zy[i, j, k], m[i, j, k+1], m[i, j-1, k],
                                 m[i, j+1, k], m[i, j-1, k+1], m[i, j+1, k+1]))

        w[i, j, k, 13] = (coeffs(d_yz[i, j, k], m[i, j+1, k], m[i, j+1, k-1],
                                 m[i, j+1, k+1], m[i, j, k-1], m[i, j, k+1]) +
                          coeffs(d_zy[i, j, k], m[i, j, k+1], m[i, j-1, k+1],
                                 m[i, j+1, k+1], m[i, j-1, k], m[i, j+1, k]))

        w[i, j, k, 14] = (coeffs(-d_xy[i, j, k], m[i+1, j, k], m[i+1, j-1, k],
                                 m[i+1, j+1, k], m[i, j-1, k], m[i, j+1, k]) +
                          coeffs(-d_yx[i, j-1, k], m[i, j-1, k], m[i-1, j-1, k],
                                 m[i+1, j-1, k], m[i-1, j, k], m[i+1, j, k]))

        w[i, j, k, 15] = (coeffs(-d_xz[i, j, k], m[i+1, j, k], m[i+1, j, k-1],
                                 m[i+1, j, k+1], m[i, j, k-1], m[i, j, k+1]) +
                          coeffs(-d_zx[i, j, k-1], m[i, j, k-1], m[i-1, j, k-1],
                                 m[i+1, j, k-1], m[i-1, j, k], m[i+1, j, k]))

        w[i, j, k, 16] = (d_x[i, j, k] * m[i+1, j, k] +
                          coeffs(-d_yx[i, j-1, k], m[i, j-1, k], m[i-1, j, k],
                                 m[i+1, j, k], m[i-1, j-1, k], m[i+1, j-1, k]) +
                          coeffs(d_yx[i, j, k], m[i, j+1, k], m[i-1, j, k],
                                 m[i+1, j, k], m[i-1, j+1, k], m[i+1, j+1, k]) +
                          coeffs(-d_zx[i, j, k-1], m[i, j, k-1], m[i-1, j, k],
                                 m[i+1, j, k], m[i-1, j, k-1], m[i+1, j, k-1]) +
                          coeffs(d_zx[i, j, k], m[i, j, k+1], m[i-1, j, k],
                                 m[i+1, j, k], m[i-1, j, k+1], m[i+1, j, k+1]))

        w[i, j, k, 17] = (coeffs(d_xz[i, j, k], m[i+1, j, k], m[i+1, j, k-1],
                                 m[i+1, j, k+1], m[i, j, k-1], m[i, j, k+1]) +
                          coeffs(d_zx[i, j, k], m[i, j, k+1], m[i-1, j, k+1],
                                 m[i+1, j, k+1], m[i-1, j, k], m[i+1, j, k]))

        w[i, j, k, 18] = (coeffs(d_xy[i, j, k], m[i+1, j, k], m[i+1, j-1, k],
                                 m[i+1, j+1, k], m[i, j-1, k], m[i, j+1, k]) +
                          coeffs(d_yx[i, j, k], m[i, j+1, k], m[i-1, j+1, k],
                                 m[i+1, j+1, k], m[i-1, j, k], m[i+1, j, k]))


class AsymmetricStencil3D(Stencil):
    def __init__(self):
        Stencil.__init__(self)

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        fibers[np.where(mesh != 1)] = 0
        weights = np.zeros((*mesh.shape, 19), dtype='float32')

        def axis_fibers(fibers, ind):
            fibr = fibers + np.roll(fibers, 1, axis=ind)
            norm = np.linalg.norm(fibr, axis=3)
            np.divide(fibr, norm[:, :, :, np.newaxis], out=fibr,
                      where=norm[:, :, :, np.newaxis] != 0)
            return fibr

        def major_diffuse(fibers, ind):
            return ((D_ac + (D_al - D_ac) * fibers[:, :, :, ind]**2) *
                    conductivity)

        def minor_diffuse(fibers, ind1, ind2):
            return (0.5 * (D_al - D_ac) * fibers[:, :, :, ind1] *
                    fibers[:, :, :, ind2] * conductivity)

        fibers_x = axis_fibers(fibers, 0)
        diffuse_x = major_diffuse(fibers_x, 0)
        diffuse_xy = minor_diffuse(fibers_x, 0, 1)
        diffuse_xz = minor_diffuse(fibers_x, 0, 2)

        fibers_y = axis_fibers(fibers, 1)
        diffuse_y = major_diffuse(fibers_y, 1)
        diffuse_yx = minor_diffuse(fibers_y, 1, 0)
        diffuse_yz = minor_diffuse(fibers_y, 1, 2)

        fibers_z = axis_fibers(fibers, 2)
        diffuse_z = major_diffuse(fibers_z, 2)
        diffuse_zx = minor_diffuse(fibers_z, 2, 0)
        diffuse_zy = minor_diffuse(fibers_z, 2, 1)

        compute_weights(weights, mesh, diffuse_x, diffuse_xy, diffuse_xz,
                        diffuse_y, diffuse_yx, diffuse_yz, diffuse_z,
                        diffuse_zx, diffuse_zy)
        weights *= dt/dr**2
        weights[:, :, :, 9] += 1

        return weights.astype('float32')