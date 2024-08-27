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
        fibers[np.where(mesh != 1)] = 0
        weights = np.zeros((*mesh.shape, 9))

        def axis_fibers(fibers, ind):
            fibr = fibers + np.roll(fibers, 1, axis=ind)
            norm = np.linalg.norm(fibr, axis=2)
            np.divide(fibr, norm[:, :, np.newaxis], out=fibr,
                      where=norm[:, :, np.newaxis] != 0)
            return fibr

        def major_diffuse(fibers, ind):
            return ((D_ac + (D_al - D_ac) * fibers[:, :, ind]**2) *
                    conductivity)

        def minor_diffuse(fibers, ind1, ind2):
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