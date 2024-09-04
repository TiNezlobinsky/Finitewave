import numpy as np
import random

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Structural3DPattern(FibrosisPattern):
    """
    A class to generate a structural fibrosis pattern in a 3D mesh grid.

    Attributes
    ----------
    x1, x2 : int
        The start and end indices for the region of interest along the x-axis.
    y1, y2 : int
        The start and end indices for the region of interest along the y-axis.
    z1, z2 : int
        The start and end indices for the region of interest along the z-axis.
    dens : float
        The density of fibrosis within the specified region, ranging from 0 (no fibrosis) to 1 (full fibrosis).
    length_i, length_j, length_k : int
        The lengths of fibrosis blocks along each axis (x, y, z).

    Methods
    -------
    generate(size, mesh=None):
        Generates a 3D mesh with a structural fibrosis pattern within the specified region.
    """

    def __init__(self, x1, x2, y1, y2, z1, z2, dens, length_i, length_j, length_k):
        """
        Initializes the Structural3DPattern object with the given region of interest, density, and block sizes.

        Parameters
        ----------
        x1, x2 : int
            The start and end indices for the region of interest along the x-axis.
        y1, y2 : int
            The start and end indices for the region of interest along the y-axis.
        z1, z2 : int
            The start and end indices for the region of interest along the z-axis.
        dens : float
            The density of fibrosis within the specified region.
        length_i, length_j, length_k : int
            The lengths of fibrosis blocks along each axis (x, y, z).
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.dens = dens
        self.length_i = length_i
        self.length_j = length_j
        self.length_k = length_k

    def generate(self, size, mesh=None):
        """
        Generates a 3D mesh with a structural fibrosis pattern within the specified region.

        If a mesh is provided, the pattern is applied to the existing mesh; otherwise, a new mesh is created.

        Parameters
        ----------
        size : tuple of int
            The size of the 3D mesh grid (x, y, z).
        mesh : numpy.ndarray, optional
            A 3D NumPy array representing the existing mesh grid to which the fibrosis pattern will be applied.
            If None, a new mesh grid of the given size is created.

        Returns
        -------
        numpy.ndarray
            A 3D NumPy array of the same size as the input, with the structural fibrosis pattern applied.
        """
        if mesh is None:
            mesh = np.zeros(size)

        for i in range(self.x1, self.x2, self.length_i):
            for j in range(self.y1, self.y2, self.length_j):
                for k in range(self.z1, self.z2, self.length_k):
                    if random.random() <= self.dens:
                        i_s = min(self.length_i, self.x2 - i)
                        j_s = min(self.length_j, self.y2 - j)
                        k_s = min(self.length_k, self.z2 - k)

                        mesh[i:i+i_s, j:j+j_s, k:k+k_s] = 2

        return mesh
