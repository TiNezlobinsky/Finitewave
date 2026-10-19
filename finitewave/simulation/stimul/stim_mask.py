from finitewave.core.stimul.stim_type.stim_current import StimCurrent
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage
from finitewave.numerics.fdm.region_selection import GridMaskSelection
from finitewave.numerics.fem.region_selection import NodeMaskSelection


class _MaskSelection:
    def __init__(self, mask):
        self.mask = mask

    def select(self, tissue):
        tissue_type = tissue.meta["type"]
        if tissue_type == "Grid":
            return GridMaskSelection(self.mask).select(tissue)
        if tissue_type == "Elements":
            return NodeMaskSelection(self.mask).select(tissue)
        raise TypeError("Mask selection does not support this tissue type.")


class StimVoltageMask(StimVoltage):
    """Apply a voltage stimulus to nodes selected by a boolean mask."""

    def __init__(self, time, voltage, mask, duration=0.0):
        super().__init__(time, voltage, duration, _MaskSelection(mask))


class StimCurrentMask(StimCurrent):
    """Apply a current stimulus to nodes selected by a boolean mask."""

    def __init__(self, time, current, duration, mask):
        super().__init__(time, current, duration, _MaskSelection(mask))


__all__ = ["StimVoltageMask", "StimCurrentMask"]
