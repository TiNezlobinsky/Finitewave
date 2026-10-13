import numpy as np


class Sphere:
    """Closed spherical region in physical coordinate space."""

    def __init__(self, center, radius):
        self.center = np.asarray(center)
        self.radius = radius

        if self.center.ndim != 1:
            raise ValueError("Sphere center must be one-dimensional.")
        if radius < 0:
            raise ValueError("Sphere radius must be non-negative.")

    @property
    def dimension(self):
        return self.center.size

    def contains(self, points):
        """Return a mask for points inside or on the sphere."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(
                "Points must have shape (n_points, sphere_dimension)."
            )
        offsets = points - self.center
        squared_distances = np.sum(offsets ** 2, axis=1)
        return squared_distances <= self.radius ** 2
