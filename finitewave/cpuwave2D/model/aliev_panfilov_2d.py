import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class AlievPanfilov2D(CardiacModel):
    """
    Implementation of the Aliev-Panfilov 2D cardiac model.

    Attributes
    ----------
    v : np.ndarray
        Array for the recovery variable.
    w : np.ndarray
        Array for diffusion weights.
    D_model : float
        Model specific diffusion coefficient
    state_vars : list
        List of state variables to be saved and restored.
    npfloat : str
        Data type used for floating-point operations, default is 'float64'.
    """

    def __init__(self):
        """
        Initializes the AlievPanfilov2D instance with default parameters.
        """
        super().__init__()
        self.v = np.ndarray
        self.w = np.ndarray
        self.D_model = 1.
        self.state_vars = ["u", "v"]
        self.npfloat = 'float64'

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.v = np.zeros_like(self.u, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.v, self.cardiac_tissue.mesh,
                        self.dt)

    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue2D
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if cardiac_tissue.fibers is None:
            return IsotropicStencil2D()

        return AsymmetricStencil2D()


@njit(parallel=True)
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
    return u_new, v
