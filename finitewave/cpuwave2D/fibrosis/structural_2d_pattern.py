import numpy as np
import random

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Structural2DPattern(FibrosisPattern):
    """
    Class for generating a structural fibrosis pattern in a 2D mesh.

    The pattern consists of rectangular blocks distributed throughout a specified region of the mesh,
    with the density controlling the likelihood of each block being present.

    Attributes
    ----------
    x1 : int
        The starting x-coordinate of the area where blocks can be placed.
    x2 : int
        The ending x-coordinate of the area where blocks can be placed.
    y1 : int
        The starting y-coordinate of the area where blocks can be placed.
    y2 : int
        The ending y-coordinate of the area where blocks can be placed.
    dens : float
        The density of the fibrosis blocks, represented as a probability.
    length_i : int
        The width of each block.
    length_j : int
        The height of each block.
    """

    def __init__(self, x1, x2, y1, y2, dens, length_i, length_j):
        """
        Initializes the Structural2DPattern with the specified parameters.

        Parameters
        ----------
        x1 : int
            The starting x-coordinate of the area where blocks can be placed.
        x2 : int
            The ending x-coordinate of the area where blocks can be placed.
        y1 : int
            The starting y-coordinate of the area where blocks can be placed.
        y2 : int
            The ending y-coordinate of the area where blocks can be placed.
        dens : float
            The density of the fibrosis blocks, represented as a probability.
        length_i : int
            The width of each block.
        length_j : int
            The height of each block.
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.dens = dens
        self.length_i = length_i
        self.length_j = length_j

    def generate(self, size, mesh=None):
        """
        Generates and applies a structural fibrosis pattern to the mesh.

        The mesh is divided into blocks of size `length_i` by `length_j`, with each block having 
        a probability `dens` of being filled with fibrosis. The function ensures that blocks do not
        extend beyond the specified region.

        Parameters
        ----------
        size : tuple of int
            The size of the mesh to create if no mesh is provided.
        mesh : np.ndarray, optional
            The mesh to which the fibrosis pattern is applied. If None, a new mesh is created 
            with the given size.

        Returns
        -------
        np.ndarray
            The mesh with the applied structural fibrosis pattern.
        """
        if mesh is None:
            mesh = np.zeros(size)

        for i in range(self.x1, self.x2, self.length_i):
            for j in range(self.y1, self.y2, self.length_j):
                if random.random() <= self.dens:
                    i_s = 0
                    j_s = 0
                    if i + self.length_i <= self.x2:
                        i_s = self.length_i
                    else:
                        i_s = self.length_i - (i + self.length_i - self.x2)

                    if j + self.length_j <= self.y2:
                        j_s = self.length_j
                    else:
                        j_s = self.length_j - (j + self.length_j - self.y2)

                    mesh[i:i + i_s, j:j + j_s] = 2

        return mesh
