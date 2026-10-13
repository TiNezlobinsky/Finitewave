import numpy as np


class NodeRegionSelection:
    """Select active FEM nodes using a physical geometric region."""

    def __init__(self, region):
        self.region = region

    def select(self, tissue):
        tissue_indexes = np.asarray(tissue.tissue_indexes)
        coords = np.asarray(tissue.coords)[tissue_indexes]

        active = np.isin(tissue_indexes, np.asarray(tissue.myo_indexes))
        return np.flatnonzero(active & self.region.contains(coords))


class NodeMaskSelection:
    """Select active FEM nodes with a node-shaped boolean mask."""

    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)

    def select(self, tissue):
        if self.mask.shape != tissue.mesh.shape:
            raise ValueError("Selection mask shape must match the node mesh shape.")

        tissue_mask = self.mask.flat[np.asarray(tissue.tissue_indexes)]
        active = np.isin(
            np.asarray(tissue.tissue_indexes),
            np.asarray(tissue.myo_indexes),
        )
        return np.flatnonzero(tissue_mask & active)


__all__ = ["NodeRegionSelection", "NodeMaskSelection"]
