import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class ScarRect2DPattern(FibrosisPattern):
    """
    Class for generating a rectangular fibrosis pattern in a 2D mesh.

    Attributes
    ----------
    x1 : int
        The starting x-coordinate of the rectangular region.
    x2 : int
        The ending x-coordinate of the rectangular region.
    y1 : int
        The starting y-coordinate of the rectangular region.
    y2 : int
        The ending y-coordinate of the rectangular region.
    """

    def __init__(self, x1, x2, y1, y2):
        """
        Initializes the ScarRect2DPattern with the specified rectangular region.

        Parameters
        ----------
        x1 : int
            The starting x-coordinate of the rectangular region.
        x2 : int
            The ending x-coordinate of the rectangular region.
        y1 : int
            The starting y-coordinate of the rectangular region.
        y2 : int
            The ending y-coordinate of the rectangular region.
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def generate(self, size, mesh=None):
        """
        Generates and applies a rectangular fibrosis pattern to the mesh.

        If no mesh is provided, a new mesh of zeros with the given size is created. The method 
        generates a rectangular region of fibrosis and applies it to the mesh.

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
            The mesh with the applied rectangular fibrosis pattern.
        """
        if mesh is None:
            mesh = np.zeros(size)

        # Apply the rectangular fibrosis pattern to the mesh
        mesh[self.x1:self.x2, self.y1:self.y2] = 2

        return mesh
