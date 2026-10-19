import numpy as np

from finitewave.core.stimul.stim_type.stim_current import StimCurrent
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage
from finitewave.geometry import Sphere
from finitewave.simulation.stimul.region_selection import (
    PhysicalRegionSelection,
    PhysicalRegionsSelection,
)


class StimVoltageSphere(StimVoltage):
    """Apply a voltage stimulus inside a sphere in physical coordinates."""

    def __init__(self, time, voltage, center, radius, duration=0.0):
        selection = PhysicalRegionSelection(Sphere(center, radius))
        super().__init__(time, voltage, duration, selection)


class StimCurrentSphere(StimCurrent):
    """Apply a current stimulus inside a sphere in physical coordinates."""

    def __init__(self, time, current, duration, center, radius):
        selection = PhysicalRegionSelection(Sphere(center, radius))
        super().__init__(time, current, duration, selection)


def _electrode_selection(coords, size):
    coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = coords[None, :]
    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError("coords must contain at least one electrode center.")
    return PhysicalRegionsSelection(Sphere(center, size) for center in coords)


class StimVoltageElectrodes(StimVoltage):
    """Apply voltage inside spherical regions around electrode centers."""

    def __init__(self, time, volt_value, coords, size):
        super().__init__(time, volt_value, selection=_electrode_selection(coords, size))


class StimCurrentElectrodes(StimCurrent):
    """Apply current inside spherical regions around electrode centers."""

    def __init__(self, time, curr_value, duration, coords, size):
        super().__init__(
            time,
            curr_value,
            duration,
            selection=_electrode_selection(coords, size),
        )


__all__ = [
    "StimVoltageSphere",
    "StimCurrentSphere",
    "StimVoltageElectrodes",
    "StimCurrentElectrodes",
]
