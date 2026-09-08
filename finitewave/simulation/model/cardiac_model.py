"""Spatial cardiac reaction model and backend-kernel integration."""

import numpy as np
from warnings import warn

from finitewave.core.model.computational_model import ComputationalModel
from .single_cell_model import SingleCellModel


class CardiacModel(ComputationalModel):
    """Backend-independent cardiac reaction model.

    The model loads its equations from a Finitewave model plugin, allocates
    state arrays over tissue points, and generates a backend-specific reaction
    kernel.

    Attributes
    ----------
    myo_indexes : array-like
        Backend array containing flat indexes of active myocytes.
    tissue_indexes : array-like
        Backend array containing flat indexes of all tissue points represented
        by the compact model arrays.
    backend : Backend
        Computational backend selected by the simulation.
    model_kernel : callable
        Backend-generated kernel that evaluates the reaction model and updates
        non-voltage state variables.
    kernel_arg_names : list of str
        Names of model values passed to ``model_kernel``.
    model_kernel_args : list
        Backend-wrapped values passed to ``model_kernel``.
    """

    def __init__(self):
        """Initialize model metadata and load the configured model plugin."""
        super().__init__()
        self.myo_indexes = None
        self.tissue_indexes = None
        self.D_model = None

        self.model_kernel_args = []
        self.model_kernel_arg_names = []
        self.array_names = []
        self.observers = []

    def __getattr__(self, name):
        state_vars = self.__dict__.get("state_vars", [])
        state_pars = self.__dict__.get("state_pars", [])

        if name in state_vars or name == "rhs":
            vals = self.__dict__.get(f"_{name}", None)
            if vals is None:
                return None

            if np.asarray(vals).size == 1:
                return np.asarray(vals).item()

            return self._unravel_array(vals)

        if name in state_pars:
            vals = self.__dict__.get(f"_{name}", None)
            if vals is None:
                return None

            if np.asarray(vals).size == 1:
                return np.asarray(vals).item()

            return self._unravel_array(vals)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        state_vars = self.__dict__.get("state_vars", [])
        state_pars = self.__dict__.get("state_pars", [])
        is_initialized = self.__dict__.get("simulation", None) is not None

        if name in state_vars and not is_initialized:
            raise AttributeError(f"Cannot set model state variable '{name}' before initialization. Use 'init_{name}' instead.")

        if name == "rhs":
            raise AttributeError("Cannot set model reaction term 'rhs' directly.")

        if (name in state_vars and is_initialized):
            np_value = np.asarray(value)
            if np_value.size == 1:
                raise AttributeError(f"Cannot set model variable '{name}' to a scalar value.")
            if np_value.size > 1:
                self.__dict__[f"_{name}"] = self._ravel_array(np_value)
                self.collect_kernel_args()
                return

        if (name in state_pars):
            np_value = np.asarray(value)

            if np_value.size == 1:
                self.__dict__[f"_{name}"] = np_value.item()

            if np_value.size > 1 and is_initialized:
                self.__dict__[f"_{name}"] = self._ravel_array(np_value)
           
            if is_initialized:
                self.generate_model_kernel()
                self.collect_kernel_args()
                return

            self.__dict__[f"_{name}"] = np_value
            return

        super().__setattr__(name, value)

    def initialize(self, simulation):
        """Allocate state and generate the reaction kernel for a simulation.

        Parameters
        ----------
        simulation : Simulation
            Simulation providing the tissue and computational backend.
        """
        self.simulation = simulation
        self.wrap_indexes()
        self.wrap_arrays()
        self.generate_model_kernel()
        self.collect_kernel_args()

    def run(self):
        """Evaluate the reaction model for one simulation time step."""
        res = self.model_kernel(
            self.simulation.dt,
            self.myo_indexes,
            self._rhs,
            self._u,
            *self.model_kernel_args,
        )
        self._reset_state_variables(res)

    def sync_backend(self):
        self.simulation.backend.sync(self._u, self._rhs, *self.model_kernel_args)

    def wrap_indexes(self):
        """Convert tissue and myocyte indexes to backend arrays."""
        tissue = self.simulation.cardiac_tissue
        backend = self.simulation.backend
        self.myo_indexes = backend.wrap_indexes(tissue.myo_indexes)
        self.tissue_indexes = backend.wrap_indexes(tissue.tissue_indexes)

    def wrap_arrays(self):
        """Allocate state and parameter arrays for represented tissue points.
        
        Parameters
        ----------
        tissue : Tissue
            The tissue for which to allocate arrays.
        backend : Backend
            The backend to use for array allocation.
        """
        self._rhs = self._ravel_array(0.)
        # allocate state arrays
        for name in self.state_vars:
            init_vals = getattr(self, f"init_{name}")
            arr = self._ravel_array(init_vals)
            setattr(self, f"_{name}", arr)
                
        # validate parameter fields shapes if they are arrays
        for name in self.state_pars:
            par = getattr(self, f"_{name}")

            if hasattr(par, '__array_namespace__') and par.size > 1:
                arr = self._ravel_array(par)
                setattr(self, f"_{name}", arr)
    
    def prepacing(self, stim_prepacing, history=False):
        """Compute initial conditions by pacing a single-cell model.
        
        Parameters
        ----------
        stim_prepacing : StimSingleCell
            Single-cell stimulation containing the time step and current trace.
        history : bool, optional
            If True, store pacing times, stimuli, and voltage history. Default
            is False.
        """

        cell_model = SingleCellModel()
        cell_model.cardiac_model = self
        cell_model.stim_sequence = stim_prepacing
        state_vars = cell_model.run(history)

        if history:
            self.pacing_times = cell_model.times
            self.pacing_stims = cell_model.stim_current
            self.u_pacing = cell_model.u_history

        # update initial conditions with the final state after prepacing
        self.set_variables(state_vars)

    def collect_kernel_args(self):
        """Collect and validate arguments required by ``model_kernel``.

        Model arrays are converted to backend arrays. Non-scalar arrays must
        contain one value per represented tissue point.

        Returns
        -------
        list
            Backend-wrapped kernel arguments in ``model_kernel_args`` order.

        Raises
        ------
        ValueError
            If an argument is uninitialized or has an incompatible size.
        """

        model_kernel_args = []

        for name in self.kernel_arg_names:
            val = getattr(self, f"_{name}", None)

            if val is None:
                raise ValueError(f"Model kernel argument '{name}' is not initialized.")

            model_kernel_args.append(val)

        self.model_kernel_args = model_kernel_args
        return model_kernel_args

    def generate_model_kernel(self):
        """Generate the backend-specific reaction kernel."""
        self.model_kernel, self.kernel_arg_names = (
            self.simulation.backend.model_generator.generate_model_kernel(self)
        )
    
    def _reset_state_variables(self, new_values):
        """Store the reaction term and state values returned by the kernel."""
        self._rhs = new_values[0]
        for i, name in enumerate(self.state_vars):
            if name == "u":
                continue
            self.__dict__[f"_{name}"] = new_values[i]
            self.model_kernel_args[i-1] = new_values[i]

    def _ravel_array(self, vals):
        """Allocate a backend array for a model variable or parameter.

        Parameters
        ----------
        name : str
            Name of the model variable or parameter.
        vals : array-like
            The array to allocate on the backend.
        tissue : Tissue
            The tissue for which to allocate arrays.
        backend : Backend
            The backend to use for array allocation.
        """
        tissue = self.simulation.cardiac_tissue
        backend = self.simulation.backend

        arr = np.asarray(vals)
        tissue_size = len(tissue.tissue_indexes)

        if arr.size == 1:
            arr = vals * np.ones((tissue_size,), dtype=np.float32)
            arr = backend.wrap_array(arr)
            return arr

        if arr.shape == tissue.tissue_indexes.shape:
            arr = backend.wrap_array(np.asarray(vals))
            return arr

        if arr.shape == tissue.mesh.shape:
            arr = backend.wrap_array(arr.ravel()[tissue.tissue_indexes])
            return arr

        raise ValueError(
            f"Array shape {arr.shape} is incompatible with tissue." +
            f" Must be scalar, {tissue.tissue_indexes.shape}, or {tissue.mesh.shape}."
        )

    def _unravel_array(self, var_data):
        """Expand a compact model array to the full tissue mesh shape.

        Locations outside the represented tissue are filled with ``NaN``.

        Parameters
        ----------
        var_data : array-like
            Compact model array containing values for represented tissue points.

        Returns
        -------
        np.ndarray
            Full-size array with the same shape as the tissue mesh.
        """
        mesh = self.simulation.cardiac_tissue.mesh
        tissue_indexes = np.asarray(self.tissue_indexes)

        if var_data.size == mesh.size:
            return np.asarray(var_data.reshape(mesh.shape))

        if var_data.size == tissue_indexes.size:
            if mesh.size / tissue_indexes.size > 1.5:
                warn(
                    "The number of tissue points is much smaller than the total mesh size. " +
                    "To reduce memory usage, consider using a ``output`` method that returns " +
                    "only the tissue points instead of the full mesh."
                )
            np_data = np.asarray(var_data)
            var_mesh = np.full_like(mesh, np.nan, dtype=np_data.dtype)
            var_mesh.flat[tissue_indexes] = np_data
            return var_mesh

        raise ValueError(
            f"Array size {var_data.size} is incompatible with tissue." +
            f" Must be {tissue_indexes.size} or {mesh.size}."
        )
    
    def output(self, name="u"):
        """Return a tissue sized flat array of the requested model variable or parameter.
        """
        var_data = getattr(self, f"_{name}", None)

        if var_data is None:
            raise ValueError(f"Variable '{name}' not found in the model.")

        np_data = np.asarray(var_data)

        if np_data.size == 1:
            return np_data.item()

        return np_data
