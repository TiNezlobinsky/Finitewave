from numba import njit, prange

from finitewave.core.exception.exceptions import IncorrectWeightsShapeError
from finitewave.cpuwave2D.model.diffuse_kernels_2d \
    import diffuse_kernel_2d_iso, diffuse_kernel_2d_aniso, _parallel


@njit(parallel=_parallel)
def ionic_kernel_2d(u_new, u, v, mesh, dt):
    a = 0.1
    k_ = 8.
    eap = 0.01
    mu_1 = 0.2
    mu_2 = 0.3

    n_i = u.shape[0]
    n_j = u.shape[1]

    for ii in prange(n_i*n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if mesh[i, j] != 1:
            continue

        v[i, j] += (- dt * (eap + (mu_1 * v[i, j]) / (mu_2 + u[i, j])) *
                    (v[i, j] + k_ * u[i, j] * (u[i, j] - a - 1.)))

        u_new[i, j] += dt * (- k_ * u[i, j] * (u[i, j] - a) * (u[i, j] - 1.) -
                             u[i, j] * v[i, j])


class AlievPanfilovKernels2D:
    def __init__(self):
        pass

    @staticmethod
    def get_diffuse_kernel(shape):
        if shape[-1] == 5:
            return diffuse_kernel_2d_iso
        if shape[-1] == 9:
            return diffuse_kernel_2d_aniso
        else:
            raise IncorrectWeightsShapeError(shape, 5, 9)

    @staticmethod
    def get_ionic_kernel():
        return ionic_kernel_2d
