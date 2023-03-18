import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


@njit
def coeffs(m0, m1, m2, m3):
    return m0 * m1 / (1 + m0 * m1 * m2 * m3)


@njit
def compute_weights(w, m, d_x, d_xy, d_y, d_yx):
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
    def __init__(self):
        Stencil.__init__(self)

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        weights = np.zeros((*mesh.shape, 9))

        fibers_x = fibers + np.roll(fibers, 1, axis=0)
        fibers_x = fibers_x / np.linalg.norm(fibers_x, axis=2)[:, :, np.newaxis]
        fibers_y = fibers + np.roll(fibers, 1, axis=1)
        fibers_y = fibers_y / np.linalg.norm(fibers_y, axis=2)[:, :, np.newaxis]

        diffuse_x = ((D_ac + (D_al - D_ac) * fibers_x[:, :, 0]**2) *
                     conductivity)
        diffuse_xy = (0.5 * (D_al - D_ac) * fibers_x[:, :, 0] *
                      fibers_x[:, :, 1] * conductivity)
        diffuse_y = ((D_ac + (D_al - D_ac) * fibers_y[:, :, 1]**2) *
                     conductivity)
        diffuse_yx = (0.5 * (D_al - D_ac) * fibers_y[:, :, 0] *
                      fibers_y[:, :, 1] * conductivity)

        compute_weights(weights, mesh, diffuse_x, diffuse_xy, diffuse_y,
                        diffuse_yx)
        weights *= dt/dr**2
        weights[:, :, 4] += 1

        return weights
