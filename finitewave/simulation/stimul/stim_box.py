from finitewave.core.stimul.stim_type.stim_current import StimCurrent
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage
from finitewave.geometry import PhysicalBox
from finitewave.numerics.fdm.region_selection import GridBox, GridRegionSelection
from finitewave.numerics.fem.region_selection import NodeRegionSelection


def _bounds(x_min, x_max, y_min, y_max, z_min, z_max):
    if (z_min is None) != (z_max is None):
        raise ValueError("z_min and z_max must either both be set or both be omitted.")
    if z_min is None:
        return (x_min, y_min), (x_max, y_max)
    return (x_min, y_min, z_min), (x_max, y_max, z_max)


class _PhysicalBoxSelection:
    def __init__(self, box):
        self.box = box

    def select(self, tissue):
        tissue_type = tissue.meta["type"]
        if tissue_type == "Grid":
            return GridRegionSelection(self.box).select(tissue)
        if tissue_type == "Elements":
            return NodeRegionSelection(self.box).select(tissue)
        raise TypeError("PhysicalBox selection does not support this tissue type.")


class StimVoltageBox(StimVoltage):
    """Voltage stimulus in a half-open box of physical coordinates."""

    def __init__(self, time, voltage, x_min, x_max, y_min, y_max,
                 z_min=None, z_max=None, duration=0.0):
        lower, upper = _bounds(
            x_min, x_max, y_min, y_max, z_min, z_max
        )
        selection = _PhysicalBoxSelection(PhysicalBox(lower, upper))
        super().__init__(time, voltage, duration, selection)


class StimCurrentBox(StimCurrent):
    """Current stimulus in a half-open box of physical coordinates."""

    def __init__(self, time, current, duration, x_min, x_max, y_min, y_max,
                 z_min=None, z_max=None):
        lower, upper = _bounds(
            x_min, x_max, y_min, y_max, z_min, z_max
        )
        selection = _PhysicalBoxSelection(PhysicalBox(lower, upper))
        super().__init__(time, current, duration, selection)


class StimVoltageGridBox(StimVoltage):
    """Voltage stimulus in a half-open box of FDM grid indexes."""

    def __init__(self, time, voltage, x_min, x_max, y_min, y_max,
                 z_min=None, z_max=None, duration=0.0):
        lower, upper = _bounds(
            x_min, x_max, y_min, y_max, z_min, z_max
        )
        super().__init__(time, voltage, duration, GridBox(lower, upper))


class StimCurrentGridBox(StimCurrent):
    """Current stimulus in a half-open box of FDM grid indexes."""

    def __init__(self, time, current, duration, x_min, x_max, y_min, y_max,
                 z_min=None, z_max=None):
        lower, upper = _bounds(
            x_min, x_max, y_min, y_max, z_min, z_max
        )
        super().__init__(time, current, duration, GridBox(lower, upper))


__all__ = [
    "StimVoltageBox",
    "StimCurrentBox",
    "StimVoltageGridBox",
    "StimCurrentGridBox",
]
