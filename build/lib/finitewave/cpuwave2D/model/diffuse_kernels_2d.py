from numba import njit, prange

_parallel = False


@njit(parallel=_parallel)
def diffuse_kernel_2d_iso(u_new, u, w, mesh):
    n_i = u.shape[0]
    n_j = u.shape[1]
    for ii in prange(n_i*n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if mesh[i, j] != 1:
            continue

        u_new[i, j] = (u[i-1, j] * w[i, j, 0] + u[i, j-1] * w[i, j, 1] +
                       u[i, j] * w[i, j, 2] + u[i, j+1] * w[i, j, 3] +
                       u[i+1, j] * w[i, j, 4])


@njit(parallel=_parallel)
def diffuse_kernel_2d_aniso(u_new, u, w, mesh):
    n_i = u.shape[0]
    n_j = u.shape[1]
    for ii in prange(n_i*n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if mesh[i, j] != 1:
            continue

        u_new[i, j] = (u[i-1, j-1] * w[i, j, 0] + u[i-1, j] * w[i, j, 1] +
                       u[i-1, j+1] * w[i, j, 2] + u[i, j-1] * w[i, j, 3] +
                       u[i, j] * w[i, j, 4] + u[i, j+1] * w[i, j, 5] +
                       u[i+1, j-1] * w[i, j, 6] + u[i+1, j] * w[i, j, 7] +
                       u[i+1, j+1] * w[i, j, 8])
