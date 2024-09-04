import numpy as np
from tqdm import tqdm

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave3D.model.tp06_3d.tp06_kernels_3d import \
    TP06Kernels3D

_npfloat = "float64"


class TP063D(CardiacModel):
    """
    A class to represent the TP06 cardiac model in 3D.

    Inherits from:
    -----------
    CardiacModel
        Base class for cardiac models.

    Attributes
    ----------
    m : np.ndarray
        Array for the gating variable m.
    h : np.ndarray
        Array for the gating variable h.
    j_ : np.ndarray
        Array for the gating variable j_.
    d : np.ndarray
        Array for the gating variable d.
    f : np.ndarray
        Array for the gating variable f.
    x : np.ndarray
        Array for the gating variable x.
    Cai_c : np.ndarray
        Array for the concentration of calcium in the intracellular space.
    model_parameters : dict
        Dictionary to hold model parameters.
    state_vars : list of str
        List of state variable names.
    npfloat : str
        Data type used for floating point operations.
    diffuse_kernel : function
        Function to handle diffusion in the model.
    ionic_kernel : function
        Function to handle ionic currents in the model.
    u : np.ndarray
        Array for membrane potential.
    u_new : np.ndarray
        Array for updated membrane potential.
    Cai : np.ndarray
        Array for calcium concentration in the intracellular space.
    CaSR : np.ndarray
        Array for calcium concentration in the sarcoplasmic reticulum.
    CaSS : np.ndarray
        Array for calcium concentration in the subsarcolemmal space.
    Nai : np.ndarray
        Array for sodium concentration in the intracellular space.
    Ki : np.ndarray
        Array for potassium concentration in the intracellular space.
    M_ : np.ndarray
        Array for gating variable M_.
    H_ : np.ndarray
        Array for gating variable H_.
    J_ : np.ndarray
        Array for gating variable J_.
    Xr1 : np.ndarray
        Array for gating variable Xr1.
    Xr2 : np.ndarray
        Array for gating variable Xr2.
    Xs : np.ndarray
        Array for gating variable Xs.
    R_ : np.ndarray
        Array for gating variable R_.
    S_ : np.ndarray
        Array for gating variable S_.
    D_ : np.ndarray
        Array for gating variable D_.
    F_ : np.ndarray
        Array for gating variable F_.
    F2_ : np.ndarray
        Array for gating variable F2_.
    FCass : np.ndarray
        Array for calcium concentration in the sarcoplasmic reticulum.
    RR : np.ndarray
        Array for calcium release from the sarcoplasmic reticulum.
    OO : np.ndarray
        Array for open states of ryanodine receptors.

    Methods
    -------
    initialize():
        Initializes the model's state variables and kernels.
    run_ionic_kernel():
        Executes the ionic kernel function to update ionic currents and state variables.
    """
    def __init__(self):
        """
        Initializes the TP063D cardiac model.

        Sets up the arrays for state variables and model parameters.
        """
        CardiacModel.__init__(self)
        self.D_al = 0.154
        self.D_ac = 0.154
        self.m = np.ndarray
        self.h = np.ndarray
        self.j_ = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.Cai_c = np.ndarray
        self.model_parameters = {}
        self.state_vars = ["u", "Cai", "CaSR", "CaSS", "Nai", "Ki",
                           "M_", "H_", "J_", "Xr1", "Xr2", "Xs", "R_",
                           "S_", "D_", "F_", "F2_", "FCass", "RR", "OO"]
        self.npfloat = 'float64'

    def initialize(self):
        """
        Initializes the model's state variables and diffusion/ionic kernels.

        Sets up the initial values for membrane potential, ion concentrations,
        gating variables, and assigns the appropriate kernel functions.
        """
        super().initialize()
        weights_shape = self.cardiac_tissue.weights.shape
        shape = self.cardiac_tissue.mesh.shape
        self.kernel_diffuse = TP06Kernels3D().get_diffuse_kernel(weights_shape)
        self.kernel_vars = TP06Kernels3D().get_ionic_kernel()

        self.u = -84.5*np.ones(shape, dtype=_npfloat)
        self.u_new = self.u.copy()
        self.Cai = 0.00007*np.ones(shape, dtype=_npfloat)
        self.CaSR = 1.3*np.ones(shape, dtype=_npfloat)
        self.CaSS = 0.00007*np.ones(shape, dtype=_npfloat)
        self.Nai = 7.67*np.ones(shape, dtype=_npfloat)
        self.Ki = 138.3*np.ones(shape, dtype=_npfloat)
        self.M_ = np.zeros(shape, dtype=_npfloat)
        self.H_ = 0.75*np.ones(shape, dtype=_npfloat)
        self.J_ = 0.75*np.ones(shape, dtype=_npfloat)
        self.Xr1 = np.zeros(shape, dtype=_npfloat)
        self.Xr2 = np.ones(shape, dtype=_npfloat)
        self.Xs = np.zeros(shape, dtype=_npfloat)
        self.R_ = np.zeros(shape, dtype=_npfloat)
        self.S_ = np.ones(shape, dtype=_npfloat)
        self.D_ = np.zeros(shape, dtype=_npfloat)
        self.F_ = np.ones(shape, dtype=_npfloat)
        self.F2_ = np.ones(shape, dtype=_npfloat)
        self.FCass = np.ones(shape, dtype=_npfloat)
        self.RR = np.ones(shape, dtype=_npfloat)
        self.OO = np.zeros(shape, dtype=_npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state variables.

        This method calls the `ionic_kernel` function from the TP06Kernels3D class,
        passing in the current state of the model and the time step.
        """
        self.ionic_kernel(self.u_new, self.u, self.Cai, self.CaSR, self.CaSS,
                          self.Nai, self.Ki, self.M_, self.H_, self.J_, self.Xr1,
                          self.Xr2, self.Xs, self.R_, self.S_, self.D_, self.F_,
                          self.F2_, self.FCass, self.RR, self.OO,
                          self.cardiac_tissue.mesh, self.dt)
