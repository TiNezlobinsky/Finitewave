import numpy as np

from finitewave.core.stimul.stim import Stim
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage


class StimVoltageWaveform(Stim):
    """Apply prescribed voltage samples to one spatial voltage stimulus.

    One sample is applied per simulation time step. The first sample is used
    at ``stimulus.t`` and all samples act on the same selected nodes.

    Parameters
    ----------
    stimulus : StimVoltage
        Spatial voltage stimulus defining start time and selected nodes.
    values : array-like
        One-dimensional sequence of prescribed voltage values.
    """

    def __init__(self, stimulus, values):
        if not isinstance(stimulus, StimVoltage):
            raise TypeError("stimulus must be a StimVoltage instance.")

        values = np.asarray(values)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("values must be a non-empty one-dimensional array.")

        super().__init__(time=stimulus.t)
        self.stimulus = stimulus
        self.values = values

    def initialize(self, simulation):
        self.simulation = simulation
        self.duration = simulation.dt * (self.values.size - 1)
        self.stimulus.initialize(simulation)
        self.stim_indexes = self.stimulus.stim_indexes

    def stimulate(self, simulation):
        sample = round((simulation.t - self.t) / simulation.dt)
        if sample < 0 or sample >= self.values.size:
            return

        self.stimulus.volt_value = self.values[sample]
        self.stimulus.stimulate(simulation)


__all__ = ["StimVoltageWaveform"]
