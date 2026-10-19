import numpy as np

from finitewave.numerics.fdm.region_selection import GridRegionSelection
from finitewave.numerics.fem.region_selection import NodeRegionSelection


class PhysicalRegionSelection:
    """Select state nodes inside a physical region for grid or FEM tissue."""

    def __init__(self, region):
        self.region = region

    def select(self, tissue):
        tissue_type = tissue.meta["type"]
        if tissue_type == "Grid":
            return GridRegionSelection(self.region).select(tissue)
        if tissue_type == "Elements":
            return NodeRegionSelection(self.region).select(tissue)
        raise TypeError("Physical region selection does not support this tissue type.")


class PhysicalRegionsSelection:
    """Select the union of nodes inside several physical regions."""

    def __init__(self, regions):
        self.regions = tuple(regions)
        if not self.regions:
            raise ValueError("At least one physical region is required.")

    def select(self, tissue):
        selections = [
            PhysicalRegionSelection(region).select(tissue)
            for region in self.regions
        ]
        return np.unique(np.concatenate(selections))
