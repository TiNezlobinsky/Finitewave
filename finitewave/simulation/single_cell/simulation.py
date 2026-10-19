import numpy as np

from finitewave.numerics.backends.model.numba_single_cell_generator import (
    NumbaSingleCellGenerator,
)
from .stimulation import SingleCellStimulation


class SingleCellSimulation:
    """Run one cardiac reaction model without a spatial tissue."""

    def __init__(self, model, dt, stimulation=None):
        self.model = model
        self.dt = dt
        self.stimulation = stimulation or SingleCellStimulation(dt)
        if self.stimulation.dt != dt:
            raise ValueError("Simulation and stimulation must use the same dt.")

        # Backend selection will be moved here with the backend refactor.
        self.kernel_generator = NumbaSingleCellGenerator()

    def add_stimulation(self, n_beats, cycle_length, current, duration):
        self.stimulation.add_stimulation(
            n_beats=n_beats,
            cycle_length=cycle_length,
            current=current,
            duration=duration,
        )
        return self

    def _collect_kernel_args(self, kernel_args_order):
        kernel_args = []
        for name in kernel_args_order:
            if name in self.model.state_vars:
                kernel_args.append(getattr(self.model, f"init_{name}"))
                continue
            if name in self.model.state_pars:
                kernel_args.append(getattr(self.model, name))
                continue
            raise ValueError(
                f"Single-cell kernel argument '{name}' was not found "
                "in model state variables or parameters."
            )
        return kernel_args

    def run(self, history=True):
        stim_current = self.stimulation.stim_current
        if stim_current.size == 0:
            raise ValueError("Add single-cell stimulation before running.")

        self.kernel_generator.history = history
        kernel, kernel_args_order = self.kernel_generator.generate_model_kernel(
            self.model
        )
        kernel_args = self._collect_kernel_args(kernel_args_order)
        initial_voltage = self.model.init_u

        if history:
            self.times = self.dt * np.arange(stim_current.size)
            self.stim_current = stim_current
            self.u_history = np.zeros(stim_current.size, dtype=np.float32)
            self.u_history[0] = initial_voltage
            kernel_args = [self.u_history] + kernel_args

        state_values = kernel(
            stim_current,
            self.dt,
            initial_voltage,
            *kernel_args,
        )
        return dict(zip(self.model.state_vars, state_values))
