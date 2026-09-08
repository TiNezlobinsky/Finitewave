"""Abstract cardiac-model interface and plugin discovery."""

from abc import ABC, abstractmethod
import copy
from importlib.metadata import entry_points


class ComputationalModel(ABC):
    """Abstract base class for reaction-diffusion models.

    Attributes
    ----------
    u : ndarray
        Array representing the action potential (mV) across the tissue.
    rhs : ndarray
        Reaction term used by time integration.
    D_model : float
        Model-specific diffusion coefficient.
    """
    def __init__(self):
        """Initialize model metadata and load plugin operations when enabled."""
        self._u = None
        self._rhs = None
        self.D_model = None

    @abstractmethod
    def initialize(self, simulation):
        """Initialize the model for a simulation."""
        pass

    @abstractmethod
    def run(self):
        """Evaluate the model for one simulation time step."""
        pass

    def set_variables(self, model_vars):
        """Set initial conditions for the model.
        Removes any existing state variables.
        
        Parameters
        ----------
        model_vars : dict
            Dictionary of model variable names and their initial values.
        """
        for name, value in model_vars.items():
            setattr(self, f"init_{name}", value)
            setattr(self, f"_{name}", None)

    def set_parameters(self, model_pars):
        """Set parameters for the model.

        Parameters
        ----------
        model_pars : dict
            Dictionary of model parameter names and their values.
        """
        for name, value in model_pars.items():
            setattr(self, f"{name}", value)

    def sync_backend(self, *args):
        """Synchronize model state with the simulation backend.

        This method is intended to be overridden by subclasses that require
        backend-specific synchronization. By default, it does nothing.
        """
        pass

    def clone(self):
        """Create a deep copy of this model.

        Returns
        -------
        ComputationalModel
            Deep copy of this model instance.
        """
        return copy.deepcopy(self)

