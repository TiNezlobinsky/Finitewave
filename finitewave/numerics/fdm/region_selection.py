import operator

import numpy as np


class GridBox:
    """Half-open box expressed in regular-grid node indexes."""

    def __init__(self, lower, upper):
        try:
            self.lower = tuple(operator.index(value) for value in lower)
            self.upper = tuple(operator.index(value) for value in upper)
        except TypeError as error:
            raise TypeError("GridBox bounds must be integer indexes.") from error
        if len(self.lower) != len(self.upper):
            raise ValueError("GridBox bounds must have the same dimension.")
        if any(lower >= upper for lower, upper in zip(self.lower, self.upper)):
            raise ValueError("Each lower bound must be less than its upper bound.")

    def select(self, tissue):
        if tissue.meta["type"] != "Grid":
            raise TypeError(
                "GridBox can only be used with regular grid tissue. "
                "Use a PhysicalBox stimulation for FEM tissue."
            )
        if len(self.lower) != tissue.mesh.ndim:
            raise ValueError("GridBox dimension must match the grid dimension.")
        slices = tuple(
            slice(lower, upper)
            for lower, upper in zip(self.lower, self.upper)
        )
        return GridSliceSelection(*slices).select(tissue)


class GridSliceSelection:
    """Select active FDM nodes using ordinary NumPy slice semantics."""

    def __init__(self, *slices):
        if not slices or not all(isinstance(item, slice) for item in slices):
            raise TypeError("GridSliceSelection expects one slice per axis.")
        self.slices = slices

    def select(self, tissue):
        if len(self.slices) != tissue.mesh.ndim:
            raise ValueError("The number of slices must match the grid dimension.")

        selected = np.zeros(tissue.mesh.shape, dtype=bool)
        selected[self.slices] = True
        selected &= tissue.mesh == 1

        state_indexes = tissue.tissue_index_map.flat[np.flatnonzero(selected)]
        return np.asarray(state_indexes, dtype=np.int64)


class GridMaskSelection:
    """Select active FDM nodes with a grid-shaped boolean mask."""

    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)

    def select(self, tissue):
        if self.mask.shape != tissue.mesh.shape:
            raise ValueError("Selection mask shape must match the grid shape.")

        selected = self.mask & (tissue.mesh == 1)
        state_indexes = tissue.tissue_index_map.flat[np.flatnonzero(selected)]
        return np.asarray(state_indexes, dtype=np.int64)


class GridRegionSelection:
    """Select active FDM nodes using a region in physical coordinates."""

    def __init__(self, region):
        self.region = region

    def select(self, tissue):
        tissue_coords = tissue.tissue_coords * tissue.dr
        selected = self.region.contains(tissue_coords)
        active = tissue.mesh.flat[tissue.tissue_indexes] == 1
        return np.flatnonzero(selected & active)


__all__ = [
    "GridBox",
    "GridSliceSelection",
    "GridMaskSelection",
    "GridRegionSelection",
]
