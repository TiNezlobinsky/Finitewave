import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Diffuse3DPattern(FibrosisPattern):
    """
    A class to generate a diffuse fibrosis pattern in a 3D mesh grid.

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

    Methods
    -------
    generate(size, mesh=None):
        Generates a 3D mesh with a diffuse fibrosis pattern within the specified region.
    """

    def __init__(self, x1, x2, y1, y2, z1, z2, dens):
        """
        Initializes the Diffuse3DPattern object with the given region of interest and density.

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
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.dens = dens

    def generate(self, size, mesh=None):
        """
        Generates a 3D mesh with a diffuse fibrosis pattern within the specified region.

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
            A 3D NumPy array of the same size as the input, with the diffuse fibrosis pattern applied.
        """
        if mesh is None:
            mesh = np.zeros(size)

        msh_area = mesh[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2]
        fib_area = np.random.uniform(size=[self.x2-self.x1, self.y2-self.y1, self.z2-self.z1])
        fib_area = np.where(fib_area < self.dens, 2, msh_area)
        mesh[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2] = fib_area

        return mesh
