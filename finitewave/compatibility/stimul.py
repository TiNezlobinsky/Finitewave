from finitewave.simulation.stimul.stim_box import (
    StimCurrentGridBox,
    StimVoltageGridBox,
)


class StimVoltageCoord(StimVoltageGridBox):
    """Legacy name for voltage stimulation using FDM grid indexes."""

    def __init__(self, time, volt_value, x_min, x_max, y_min, y_max,
                 z_min=0, z_max=0):
        if z_min == 0 and z_max == 0:
            z_min = z_max = None
        super().__init__(
            time, volt_value, x_min, x_max, y_min, y_max, z_min, z_max
        )


class StimCurrentCoord(StimCurrentGridBox):
    """Legacy name for current stimulation using FDM grid indexes."""

    def __init__(self, time, curr_value, duration, x_min, x_max, y_min, y_max,
                 z_min=None, z_max=None):
        super().__init__(
            time, curr_value, duration,
            x_min, x_max, y_min, y_max, z_min, z_max
        )


__all__ = ["StimCurrentCoord", "StimVoltageCoord"]
