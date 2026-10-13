import numpy as np


class PhysicalBox:
    """Axis-aligned half-open box in physical coordinate space."""

    def __init__(self, lower, upper):
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)

        if self.lower.ndim != 1 or self.upper.ndim != 1:
            raise ValueError("PhysicalBox bounds must be one-dimensional.")
        if self.lower.shape != self.upper.shape:
            raise ValueError("PhysicalBox bounds must have the same dimension.")
        if np.any(self.lower >= self.upper):
            raise ValueError("Each lower bound must be less than its upper bound.")

    @property
    def dimension(self):
        return self.lower.size

    def contains(self, points):
        """Return a mask for points inside ``[lower, upper)``."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(
                "Points must have shape (n_points, box_dimension)."
            )
        return np.all(
            (points >= self.lower) & (points < self.upper), axis=1
        )
