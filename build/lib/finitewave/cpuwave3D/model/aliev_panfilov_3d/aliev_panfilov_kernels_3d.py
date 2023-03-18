from numba import njit, prange

from finitewave.core.exception.exceptions import IncorrectWeightsShapeError
from finitewave.cpuwave3D.model.diffuse_kernels_3d \
    import diffuse_kernel_3d_iso, diffuse_kernel_3d_aniso, _parallel


@njit(parallel=_parallel)
def ionic_kernel_3d(u_new, u, v, mesh, dt):
    # constants
    a = 0.1
    k_ = 8.
    eap = 0.01
    mu_1 = 0.2
    mu_2 = 0.3

    n_i = u.shape[0]
    n_j = u.shape[1]
    n_k = u.shape[2]

    for ii in prange(n_i*n_j*n_k):
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k
        if mesh[i, j, k] != 1:
            continue

        u_new[i, j, k] += dt * (- k_ * u[i, j, k] * (u[i, j, k] - a) *
                                (u[i, j, k] - 1.) - u[i, j, k] * v[i, j, k])

        v[i, j, k] += (- dt * (eap + (mu_1 * v[i, j, k]) / (mu_2 + u[i, j, k]))
                       * (v[i, j, k] + k_ * u[i, j, k] * (u[i, j, k] - a - 1.)))


class AlievPanfilovKernels3D:
    def __init__(self):
        pass

    @staticmethod
    def get_diffuse_kernel(shape):
        if shape[-1] == 7:
            return diffuse_kernel_3d_iso
        if shape[-1] == 19:
            return diffuse_kernel_3d_aniso
        else:
            raise IncorrectWeightsShapeError(shape, 7, 19)

    @staticmethod
    def get_ionic_kernel():
        return ionic_kernel_3d
