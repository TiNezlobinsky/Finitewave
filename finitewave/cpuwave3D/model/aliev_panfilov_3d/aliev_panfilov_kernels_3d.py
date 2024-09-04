from numba import njit, prange

from finitewave.core.exception.exceptions import IncorrectWeightsShapeError
from finitewave.cpuwave3D.model.diffuse_kernels_3d \
    import diffuse_kernel_3d_iso, diffuse_kernel_3d_aniso, _parallel


@njit(parallel=_parallel)
def ionic_kernel_3d(u_new, u, v, mesh, dt):
    """
    Computes the ionic kernel for the Aliev-Panfilov 3D model.

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
    """
    Provides kernel functions for the Aliev-Panfilov 3D model.

    This class includes methods for retrieving diffusion and ionic kernels
    specific to the Aliev-Panfilov 3D model.

    Methods
    -------
    get_diffuse_kernel(shape)
        Returns the appropriate diffusion kernel function based on the shape of weights.
    
    get_ionic_kernel()
        Returns the ionic kernel function for the Aliev-Panfilov 3D model.
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
        if shape[-1] == 7:
            return diffuse_kernel_3d_iso
        if shape[-1] == 19:
            return diffuse_kernel_3d_aniso
        else:
            raise IncorrectWeightsShapeError(shape, 7, 19)

    @staticmethod
    def get_ionic_kernel():
        """
        Retrieves the ionic kernel function for the Aliev-Panfilov 3D model.

        Returns
        -------
        function
            The ionic kernel function.
        """
        return ionic_kernel_3d
