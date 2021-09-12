import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


@njit
def coeffs(d, m0, m1, m2, m3):
    return d * m0 * m1 / (1 + m0 * m1 * m2 * m3)


@njit
def corner(d0, m00, m01, m02, m03, d1, m10, m11, m12, m13):
    return 0.5 * (coeffs(d0, m00, m01, m02, m03) +
                  coeffs(d1, m10, m11, m12, m13))


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

        w[i, j, k, 0] = corner(d_xy[i-1, j, k], m[i-1, j-1, k], m[i-1, j+1, k],
                               m[i, j-1, k], m[i, j+1, k],
                               d_yx[i, j-1, k], m[i-1, j-1, k], m[i+1, j-1, k],
                               m[i-1, j, k], m[i+1, j, k])

        w[i, j, k, 1] = corner(d_xz[i-1, j, k], m[i-1, j, k-1], m[i-1, j, k+1],
                               m[i, j, k-1], m[i, j, k+1],
                               d_zx[i, j, k-1], m[i-1, j, k-1], m[i+1, j, k-1],
                               m[i-1, j, k], m[i+1, j, k])

        w[i, j, k, 2] = (d_x[i-1, j, k] * m[i-1, j, k] +
                         corner(d_yx[i, j-1, k], m[i-1, j, k], m[i+1, j, k],
                                m[i-1, j-1, k], m[i+1, j-1, k],
                                -d_yx[i, j, k], m[i-1, j, k], m[i+1, j, k],
                                m[i-1, j+1, k], m[i+1, j+1, k]) +
                         corner(d_zx[i, j, k-1], m[i-1, j, k], m[i+1, j, k],
                                m[i-1, j, k-1], m[i+1, j, k-1],
                                -d_zx[i, j, k], m[i-1, j, k], m[i+1, j, k],
                                m[i-1, j, k+1], m[i+1, j, k+1]))

        w[i, j, k, 3] = corner(-d_xz[i-1, j, k], m[i-1, j, k-1], m[i-1, j, k+1],
                               m[i, j, k-1], m[i, j, k+1],
                               -d_zx[i, j, k], m[i-1, j, k+1], m[i+1, j, k+1],
                               m[i-1, j, k], m[i+1, j, k])

        w[i, j, k, 4] = corner(-d_xy[i-1, j, k], m[i-1, j-1, k], m[i-1, j+1, k],
                               m[i, j-1, k], m[i, j+1, k],
                               -d_yx[i, j, k], m[i-1, j+1, k], m[i+1, j+1, k],
                               m[i-1, j, k], m[i+1, j, k])

        w[i, j, k, 5] = corner(d_yz[i, j-1, k], m[i, j-1, k-1], m[i, j-1, k+1],
                               m[i, j, k-1], m[i, j, k+1],
                               d_zy[i, j, k-1], m[i, j-1, k-1], m[i, j+1, k-1],
                               m[i, j-1, k], m[i, j+1, k])

        w[i, j, k, 6] = (d_y[i, j-1, k] * m[i, j-1, k] +
                         corner(d_xy[i-1, j, k], m[i, j-1, k], m[i, j+1, k],
                                m[i-1, j-1, k], m[i-1, j+1, k],
                                -d_xy[i, j, k], m[i, j-1, k], m[i, j+1, k],
                                m[i+1, j-1, k], m[i+1, j+1, k]) +
                         corner(d_zy[i, j, k-1], m[i, j-1, k], m[i, j+1, k],
                                m[i, j-1, k-1], m[i, j+1, k-1],
                                -d_zy[i, j, k], m[i, j-1, k], m[i, j+1, k],
                                m[i, j-1, k+1], m[i, j+1, k+1]))

        w[i, j, k, 7] = corner(-d_yz[i, j-1, k], m[i, j-1, k-1], m[i, j-1, k+1],
                               m[i, j, k-1], m[i, j, k+1],
                               -d_zy[i, j, k], m[i, j-1, k+1], m[i, j+1, k+1],
                               m[i, j-1, k], m[i, j+1, k])

        w[i, j, k, 8] = (d_z[i, j, k-1] * m[i, j, k-1] +
                         corner(d_yz[i, j-1, k], m[i, j, k-1], m[i, j, k+1],
                                m[i, j-1, k-1], m[i, j-1, k+1],
                                -d_yz[i, j, k], m[i, j, k-1], m[i, j, k+1],
                                m[i, j+1, k-1], m[i, j+1, k+1]) +
                         corner(d_xz[i, j, k-1], m[i, j, k-1], m[i, j, k+1],
                                m[i-1, j, k-1], m[i-1, j, k+1],
                                -d_xz[i, j, k], m[i, j, k-1], m[i, j, k+1],
                                m[i+1, j, k-1], m[i+1, j, k+1]))

        w[i, j, k, 9] = - (m[i-1, j, k] * d_x[i-1, j, k] +
                           m[i+1, j, k] * d_x[i, j, k] +
                           m[i, j-1, k] * d_y[i, j-1, k] +
                           m[i, j+1, k] * d_y[i, j, k] +
                           m[i, j, k-1] * d_z[i, j, k-1] +
                           m[i, j, k+1] * d_z[i, j, k])

        w[i, j, k, 10] = (d_z[i, j, k] * m[i, j, k+1] +
                          corner(-d_xz[i-1, j, k], m[i, j, k-1], m[i, j, k+1],
                                 m[i-1, j, k-1], m[i-1, j, k+1],
                                 d_xz[i, j, k], m[i, j, k-1], m[i, j, k+1],
                                 m[i+1, j, k-1], m[i+1, j, k+1]) +
                          corner(-d_yz[i, j-1, k], m[i, j, k-1], m[i, j, k+1],
                                 m[i, j-1, k-1], m[i, j-1, k+1],
                                 d_yz[i, j, k], m[i, j, k-1], m[i, j, k+1],
                                 m[i, j+1, k-1], m[i, j+1, k+1]))

        w[i, j, k, 11] = corner(-d_yz[i, j, k], m[i, j+1, k-1],
                                m[i, j+1, k+1], m[i, j, k-1], m[i, j, k+1],
                                -d_zy[i, j, k-1], m[i, j-1, k-1],
                                m[i, j+1, k-1], m[i, j-1, k], m[i, j+1, k])

        w[i, j, k, 12] = (d_y[i, j, k] * m[i, j+1, k] +
                          corner(-d_xy[i-1, j, k], m[i, j-1, k], m[i, j+1, k],
                                 m[i-1, j-1, k], m[i-1, j+1, k],
                                 d_xy[i, j, k], m[i, j-1, k], m[i, j+1, k],
                                 m[i+1, j-1, k], m[i+1, j+1, k]) +
                          corner(-d_zy[i, j, k-1], m[i, j-1, k], m[i, j+1, k],
                                 m[i, j-1, k-1], m[i, j+1, k-1],
                                 d_zy[i, j, k], m[i, j-1, k], m[i, j+1, k],
                                 m[i, j-1, k+1], m[i, j+1, k+1]))

        w[i, j, k, 13] = corner(d_yz[i, j, k], m[i, j+1, k-1],
                                m[i, j+1, k+1], m[i, j, k-1], m[i, j, k+1],
                                d_zy[i, j, k], m[i, j-1, k+1],
                                m[i, j+1, k+1], m[i, j-1, k], m[i, j+1, k])

        w[i, j, k, 14] = corner(-d_xy[i, j, k], m[i+1, j-1, k],
                                m[i+1, j+1, k], m[i, j-1, k], m[i, j+1, k],
                                -d_yx[i, j-1, k], m[i-1, j-1, k],
                                m[i+1, j-1, k], m[i-1, j, k], m[i+1, j, k])

        w[i, j, k, 15] = corner(-d_xz[i, j, k], m[i+1, j, k-1],
                                m[i+1, j, k+1], m[i, j, k-1], m[i, j, k+1],
                                -d_zx[i, j, k-1], m[i-1, j, k-1],
                                m[i+1, j, k-1], m[i-1, j, k], m[i+1, j, k])

        w[i, j, k, 16] = (d_x[i, j, k] * m[i+1, j, k] +
                          corner(-d_yx[i, j-1, k], m[i-1, j, k], m[i+1, j, k],
                                 m[i-1, j-1, k], m[i+1, j-1, k],
                                 d_yx[i, j, k], m[i-1, j, k], m[i+1, j, k],
                                 m[i-1, j+1, k], m[i+1, j+1, k]) +
                          corner(-d_zx[i, j, k-1], m[i-1, j, k], m[i+1, j, k],
                                 m[i-1, j, k-1], m[i+1, j, k-1],
                                 d_zx[i, j, k], m[i-1, j, k], m[i+1, j, k],
                                 m[i-1, j, k+1], m[i+1, j, k+1]))

        w[i, j, k, 17] = corner(d_xz[i, j, k], m[i+1, j, k-1],
                                m[i+1, j, k+1], m[i, j, k-1], m[i, j, k+1],
                                d_zx[i, j, k], m[i-1, j, k+1],
                                m[i+1, j, k+1], m[i-1, j, k], m[i+1, j, k])

        w[i, j, k, 18] = corner(d_xy[i, j, k], m[i+1, j-1, k],
                                m[i+1, j+1, k], m[i, j-1, k], m[i, j+1, k],
                                d_yx[i, j, k], m[i-1, j+1, k],
                                m[i+1, j+1, k], m[i-1, j, k], m[i+1, j, k])


class AsymmetricStencil3D(Stencil):
    def __init__(self):
        Stencil.__init__(self)

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        weights = np.zeros((*mesh.shape, 19))

        fibers_x = fibers + np.roll(fibers, 1, axis=0)
        fibers_x = fibers_x / np.linalg.norm(fibers_x, axis=3)[:, :, :,
                                                               np.newaxis]
        fibers_y = fibers + np.roll(fibers, 1, axis=1)
        fibers_y = fibers_y / np.linalg.norm(fibers_y, axis=3)[:, :, :,
                                                               np.newaxis]

        fibers_z = fibers + np.roll(fibers, 1, axis=2)
        fibers_z = fibers_z / np.linalg.norm(fibers_z, axis=3)[:, :, :,
                                                               np.newaxis]

        diffuse_x = ((D_ac + (D_al - D_ac) * fibers_x[:, :, :, 0]**2) *
                     conductivity)
        diffuse_xy = (0.5 * (D_al - D_ac) * fibers_x[:, :, :, 0] *
                      fibers_x[:, :, :, 1] * conductivity)
        diffuse_xz = (0.5 * (D_al - D_ac) * fibers_x[:, :, :, 0] *
                      fibers_x[:, :, :, 2] * conductivity)

        diffuse_y = ((D_ac + (D_al - D_ac) * fibers_y[:, :, :, 1]**2) *
                     conductivity)
        diffuse_yx = (0.5 * (D_al - D_ac) * fibers_y[:, :, :, 1] *
                      fibers_y[:, :, :, 0] * conductivity)
        diffuse_yz = (0.5 * (D_al - D_ac) * fibers_y[:, :, :, 1] *
                      fibers_y[:, :, :, 2] * conductivity)

        diffuse_z = ((D_ac + (D_al - D_ac) * fibers_z[:, :, :, 2]**2) *
                     conductivity)
        diffuse_zx = (0.5 * (D_al - D_ac) * fibers_z[:, :, :, 2] *
                      fibers_z[:, :, :, 0] * conductivity)
        diffuse_zy = (0.5 * (D_al - D_ac) * fibers_z[:, :, :, 2] *
                      fibers_z[:, :, :, 1] * conductivity)

        compute_weights(weights, mesh, diffuse_x, diffuse_xy, diffuse_xz,
                        diffuse_y, diffuse_yx, diffuse_yz, diffuse_z,
                        diffuse_zx, diffuse_zy)
        weights *= dt/dr**2
        weights[:, :, :, 9] += 1

        # import matplotlib.pyplot as plt
        #
        # fig, axs = plt.subplots(3, 3)
        # for s, n in enumerate([0, 2, 4, 6, 9, 12, 11, 16, 18]):
        #     i = s // 3
        #     j = s % 3
        #     axs[i, j].imshow(weights[5, :, :, n])
        # plt.show()
        #
        # fig, axs = plt.subplots(3, 3)
        # for s, n in enumerate([0, 2, 4, 6, 9, 12, 11, 16, 18]):
        #     i = s // 3
        #     j = s % 3
        #     axs[i, j].imshow(weights[:, 5, :, n])
        # plt.show()
        #
        # fig, axs = plt.subplots(3, 3)
        # for s, n in enumerate([0, 2, 4, 6, 9, 12, 11, 16, 18]):
        #     i = s // 3
        #     j = s % 3
        #     axs[i, j].imshow(weights[:, :, 5, n])
        # plt.show()

        # fig, axs = plt.subplots(3, 3)
        # for s, n in enumerate([1, 2, 3, 8, 9, 10, 15, 16, 17]):
        #     i = s // 3
        #     j = s % 3
        #     axs[i, j].imshow(weights[:, 5, :, n])
        # plt.show()
        #
        # fig, axs = plt.subplots(3, 3)
        # for s, n in enumerate([5, 6, 7, 8, 9, 10, 11, 12, 13]):
        #     i = s // 3
        #     j = s % 3
        #     axs[i, j].imshow(weights[:, :, 5, n])
        # plt.show()

        return weights
