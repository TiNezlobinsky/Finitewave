from numba import njit, prange


@njit(parallel=True)
def aliev_panfilov_ionic_kernel_2d(u_new, u, v, mesh, dt):
    """
    Computes the ionic kernel for the Aliev-Panfilov 2D model.

    This function updates the action potential (u) and recovery variable (v) 
    based on the Aliev-Panfilov model equations.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated action potential values.
    u : np.ndarray
        Current action potential array.
    v : np.ndarray
        Recovery variable array.
    mesh : np.ndarray
        Tissue mesh array indicating tissue types.
    dt : float
        Time step for the simulation.
    """
    a = 0.1
    k_ = 8.0
    eap = 0.01
    mu_1 = 0.2
    mu_2 = 0.3

    n_i = u.shape[0]
    n_j = u.shape[1]

    for ii in prange(n_i * n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if mesh[i, j] != 1:
            continue

        v[i, j] += (- dt * (eap + (mu_1 * v[i, j]) / (mu_2 + u[i, j])) *
                    (v[i, j] + k_ * u[i, j] * (u[i, j] - a - 1.)))

        u_new[i, j] += dt * (- k_ * u[i, j] * (u[i, j] - a) * (u[i, j] - 1.) -
                             u[i, j] * v[i, j])
