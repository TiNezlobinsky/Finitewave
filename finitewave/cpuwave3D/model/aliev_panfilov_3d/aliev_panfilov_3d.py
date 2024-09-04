import numpy as np
from tqdm import tqdm

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave3D.model.aliev_panfilov_3d.aliev_panfilov_kernels_3d import \
    AlievPanfilovKernels3D

_npfloat = "float64"


class AlievPanfilov3D(CardiacModel):
    """
    Implementation of the Aliev-Panfilov 3D cardiac model.

    This model simulates the electrical activity in cardiac tissue using the 
    Aliev-Panfilov equations. It extends the CardiacModel base class and provides
    methods to initialize the model, run the ionic kernel, and handle simulation state.

    Attributes
    ----------
    v : np.ndarray
        Array for the recovery variable.
    w : np.ndarray
        Array for diffusion weights.
    state_vars : list
        List of state variables to be saved and restored.
    npfloat : str
        Data type used for floating-point operations, default is 'float64'.
    diffuse_kernel : function
        Function for performing diffusion computations.
    ionic_kernel : function
        Function for performing ionic computations.
    """
    def __init__(self):
        CardiacModel.__init__(self)
        self.v = np.ndarray
        self.w = np.ndarray
        self.state_vars = ["u", "v"]
        self.npfloat = _npfloat

    def initialize(self):
        """
        Initializes the model for simulation.

        This method sets up the diffusion and ionic kernel functions, initializes
        arrays for the action potential and recovery variable, and prepares the model 
        for simulation. It calls the base class initialization method and sets up 
        the diffusion and ionic kernels specific to the Aliev-Panfilov model.
        """
        super().initialize()
        weights_shape = self.cardiac_tissue.weights.shape
        shape = self.cardiac_tissue.mesh.shape
        self.diffuse_kernel = AlievPanfilovKernels3D().get_diffuse_kernel(weights_shape)
        self.ionic_kernel = AlievPanfilovKernels3D().get_ionic_kernel()
        self.v = np.zeros(shape, dtype=_npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.

        This method updates the action potential and recovery variable arrays using
        the ionic kernel function retrieved during initialization.

        It applies the Aliev-Panfilov equations to compute the next state of the 
        action potential and recovery variable based on the current state of the model.
        """
        self.ionic_kernel(self.u_new, self.u, self.v, self.cardiac_tissue.mesh,
                          self.dt)
