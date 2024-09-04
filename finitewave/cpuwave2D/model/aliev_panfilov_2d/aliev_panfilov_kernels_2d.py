from numba import njit, prange

from finitewave.core.exception.exceptions import IncorrectWeightsShapeError
from finitewave.cpuwave2D.model.diffuse_kernels_2d import diffuse_kernel_2d_iso, diffuse_kernel_2d_aniso, _parallel


@njit(parallel=_parallel)
def ionic_kernel_2d(u_new, u, v, mesh, dt):
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


class AlievPanfilovKernels2D:
    """
    Provides kernel functions for the Aliev-Panfilov 2D model.

    This class includes methods for retrieving diffusion and ionic kernels
    specific to the Aliev-Panfilov 2D model.

    Methods
    -------
    get_diffuse_kernel(shape)
        Returns the appropriate diffusion kernel function based on the shape of weights.
    
    get_ionic_kernel()
        Returns the ionic kernel function for the Aliev-Panfilov 2D model.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_diffuse_kernel(shape):
        """
        Retrieves the diffusion kernel function based on the shape of weights.

        Parameters
        ----------
        shape : tuple
            The shape of the weights array used for determining the diffusion kernel.

        Returns
        -------
        function
            The appropriate diffusion kernel function.

        Raises
        ------
        IncorrectWeightsShapeError
            If the shape of the weights array is not recognized.
        """
        if shape[-1] == 5:
            return diffuse_kernel_2d_iso
        if shape[-1] == 9:
            return diffuse_kernel_2d_aniso
        else:
            raise IncorrectWeightsShapeError(shape, 5, 9)

    @staticmethod
    def get_ionic_kernel():
        """
        Retrieves the ionic kernel function for the Aliev-Panfilov 2D model.

        Returns
        -------
        function
            The ionic kernel function.
        """
        return ionic_kernel_2d
