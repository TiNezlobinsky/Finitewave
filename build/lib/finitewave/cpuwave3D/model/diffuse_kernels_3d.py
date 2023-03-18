from numba import njit, prange

_parallel = False


@njit(parallel=_parallel)
def diffuse_kernel_3d_iso(u_new, u, w, mesh):
    n_i = u.shape[0]
    n_j = u.shape[1]
    n_k = u.shape[2]
    for ii in prange(n_i*n_j*n_k):
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k
        if mesh[i, j, k] != 1:
            continue

        u_new[i, j, k] = (u[i-1, j, k] * w[i, j, k, 0] +
                          u[i, j-1, k] * w[i, j, k, 1] +
                          u[i, j, k-1] * w[i, j, k, 2] +
                          u[i, j, k] * w[i, j, k, 3] +
                          u[i, j, k+1] * w[i, j, k, 4] +
                          u[i, j+1, k] * w[i, j, k, 5] +
                          u[i+1, j, k] * w[i, j, k, 6])


@njit(parallel=_parallel)
def diffuse_kernel_3d_aniso(u_new, u, w, mesh):
    n_i = u.shape[0]
    n_j = u.shape[1]
    n_k = u.shape[2]
    for ii in prange(n_i*n_j*n_k):
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k
        if mesh[i, j, k] != 1:
            continue

        u_new[i, j, k] = (u[i-1, j-1, k] * w[i, j, k, 0] +
                          u[i-1, j, k-1] * w[i, j, k, 1] +
                          u[i-1, j, k] * w[i, j, k, 2] +
                          u[i-1, j, k+1] * w[i, j, k, 3] +
                          u[i-1, j+1, k] * w[i, j, k, 4] +
                          u[i, j-1, k-1] * w[i, j, k, 5] +
                          u[i, j-1, k] * w[i, j, k, 6] +
                          u[i, j-1, k+1] * w[i, j, k, 7] +
                          u[i, j, k-1] * w[i, j, k, 8] +
                          u[i, j, k] * w[i, j, k, 9] +
                          u[i, j, k+1] * w[i, j, k, 10] +
                          u[i, j+1, k-1] * w[i, j, k, 11] +
                          u[i, j+1, k] * w[i, j, k, 12] +
                          u[i, j+1, k+1] * w[i, j, k, 13] +
                          u[i+1, j-1, k] * w[i, j, k, 14] +
                          u[i+1, j, k-1] * w[i, j, k, 15] +
                          u[i+1, j, k] * w[i, j, k, 16] +
                          u[i+1, j, k+1] * w[i, j, k, 17] +
                          u[i+1, j+1, k] * w[i, j, k, 18])
