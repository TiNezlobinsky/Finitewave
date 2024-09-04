from numba import njit, prange

_parallel = False


@njit(parallel=_parallel)
def diffuse_kernel_3d_iso(u_new, u, w, mesh):
    """
    Performs isotropic diffusion on a 3D grid.

    This function computes the new values of the potential field `u_new` based on an isotropic 
    diffusion model. The computation is performed in parallel using Numba's JIT compilation.

    Parameters
    ----------
    u_new : numpy.ndarray
        A 3D array to store the updated potential values after diffusion.
    
    u : numpy.ndarray
        A 3D array representing the current potential values before diffusion.
    
    w : numpy.ndarray
        A 4D array of weights used in the diffusion computation. The shape should match (n_i, n_j, n_k, 7),
        where `n_i`, `n_j` and `n_k` are the dimensions of the `u` and `u_new` arrays. 
    
    mesh : numpy.ndarray
        A 3D array representing the mesh of the tissue. Each element indicates the type of tissue at
        that position (e.g., cardiomyocyte, empty, or fibrosis). Only positions with a value of 1 are
        considered for diffusion.

    Notes
    -----
    The diffusion is applied only to points in the `mesh` with a value of 1. Boundary conditions are
    not explicitly handled and are assumed to be implicitly managed by the provided mesh.
    """
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
    """
    Performs anisotropic diffusion on a 3D grid.

    This function computes the new values of the potential field `u_new` based on an anisotropic 
    diffusion model. The computation is performed in parallel using Numba's JIT compilation.

    Parameters
    ----------
    u_new : numpy.ndarray
        A 3D array to store the updated potential values after diffusion.
    
    u : numpy.ndarray
        A 3D array representing the current potential values before diffusion.
    
    w : numpy.ndarray
        A 4D array of weights used in the diffusion computation. The shape should match (n_i, n_j, n_k, 19),
        where `n_i`, `n_j` and `n_k` are the dimensions of the `u` and `u_new` arrays.
    
    mesh : numpy.ndarray
        A 3D array representing the mesh of the tissue. Each element indicates the type of tissue at
        that position (e.g., cardiomyocyte, empty, or fibrosis). Only positions with a value of 1 are
        considered for diffusion.

    Notes
    -----
    The diffusion is applied only to points in the `mesh` with a value of 1. Boundary conditions are
    not explicitly handled and are assumed to be implicitly managed by the provided mesh.
    """
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
