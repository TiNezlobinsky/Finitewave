import numpy as np
from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.model.luo_rudy91_2d.luo_rudy91_kernels_2d import LuoRudy91Kernels2D

_npfloat = "float64"

class LuoRudy912D(CardiacModel):
    """
    Implements the 2D Luo-Rudy 1991 cardiac model for simulating cardiac electrical activity.

    This class initializes the state variables and provides methods for running simulations with the Luo-Rudy 1991 model.

    Attributes
    ----------
    m : np.ndarray
        Gating variable m.
    h : np.ndarray
        Gating variable h.
    j_ : np.ndarray
        Gating variable j_.
    d : np.ndarray
        Gating variable d.
    f : np.ndarray
        Gating variable f.
    x : np.ndarray
        Gating variable x.
    Cai_c : np.ndarray
        Intracellular calcium concentration.
    model_parameters : dict
        Dictionary to hold model-specific parameters.
    state_vars : list
        List of state variable names.
    npfloat : str
        NumPy data type used for floating point calculations ('float64').

    Methods
    -------
    initialize():
        Initializes the state variables and sets up the diffusion and ionic kernels.
    run_ionic_kernel():
        Executes the ionic kernel to update the state variables and membrane potential.
    """

    def __init__(self):
        """
        Initializes the LuoRudy912D instance, setting up the state variables and parameters.
        """
        CardiacModel.__init__(self)
        self.m = np.ndarray
        self.h = np.ndarray
        self.j_ = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.Cai_c = np.ndarray
        self.model_parameters = {}
        self.state_vars = ["u", "m", "h", "j_", "d", "f", "x", "Cai_c"]
        self.npfloat = 'float64'

    def initialize(self):
        """
        Initializes the state variables and sets up the diffusion and ionic kernels.

        This method sets the initial values for the membrane potential `u`, gating variables `m`, `h`, `j_`, `d`, `f`, `x`, 
        and intracellular calcium concentration `Cai_c`. It also retrieves and sets the diffusion and ionic kernel functions
        based on the shape of the weights in the cardiac tissue.
        """
        super().initialize()
        weights_shape = self.cardiac_tissue.weights.shape
        shape = self.cardiac_tissue.mesh.shape

        self.diffuse_kernel = LuoRudy91Kernels2D().get_diffuse_kernel(weights_shape)
        self.ionic_kernel = LuoRudy91Kernels2D().get_ionic_kernel()

        self.u = -84.5 * np.ones(shape, dtype=_npfloat)
        self.u_new = self.u.copy()
        self.m = 0.0017 * np.ones(shape, dtype=_npfloat)
        self.h = 0.9832 * np.ones(shape, dtype=_npfloat)
        self.j_ = 0.995484 * np.ones(shape, dtype=_npfloat)
        self.d = 0.000003 * np.ones(shape, dtype=_npfloat)
        self.f = np.ones(shape, dtype=_npfloat)
        self.x = 0.0057 * np.ones(shape, dtype=_npfloat)
        self.Cai_c = 0.0002 * np.ones(shape, dtype=_npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel to update the state variables and membrane potential.

        This method calls the ionic kernel function provided by the `LuoRudy91Kernels2D` class to compute the updates for
        the membrane potential `u_new` and the gating variables `m`, `h`, `j_`, `d`, `f`, `x`, and `Cai_c` based on the
        current state and the time step `dt`.

        The ionic kernel function takes the following parameters:
        - `u_new`: Array to store updated membrane potential values.
        - `u`: Array of current membrane potential values.
        - `m`: Array of gating variable m.
        - `h`: Array of gating variable h.
        - `j_`: Array of gating variable j_.
        - `d`: Array of gating variable d.
        - `f`: Array of gating variable f.
        - `x`: Array of gating variable x.
        - `Cai_c`: Array of intracellular calcium concentration.
        - `mesh`: Array indicating tissue types.
        - `dt`: Time step for the simulation.
        """
        self.ionic_kernel(self.u_new, self.u, self.m, self.h, self.j_, self.d,
                          self.f, self.x, self.Cai_c, self.cardiac_tissue.mesh,
                          self.dt)
