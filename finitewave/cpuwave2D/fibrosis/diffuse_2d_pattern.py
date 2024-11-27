import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Diffuse2DPattern(FibrosisPattern):
    """
    Class for generating a diffuse 2D fibrosis pattern in a given mesh area.

    Attributes
    ----------
    x1 : int
        The starting x-coordinate of the fibrosis area.
    x2 : int
        The ending x-coordinate of the fibrosis area.
    y1 : int
        The starting y-coordinate of the fibrosis area.
    y2 : int
        The ending y-coordinate of the fibrosis area.
    dens : float
        The density of the fibrosis, where a value between 0 and 1 represents the probability 
        of fibrosis in each cell of the specified area.
    """

    def __init__(self, x1, x2, y1, y2, dens):
        """
        Initializes the Diffuse2DPattern with the specified parameters.

        Parameters
        ----------
        x1 : int
            The starting x-coordinate of the fibrosis area.
        x2 : int
            The ending x-coordinate of the fibrosis area.
        y1 : int
            The starting y-coordinate of the fibrosis area.
        y2 : int
            The ending y-coordinate of the fibrosis area.
        dens : float
            The density of the fibrosis, where a value between 0 and 1 represents the probability 
            of fibrosis in each cell of the specified area.
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.dens = dens

    def generate(self, size, mesh=None):
        """
        Generates and applies the diffuse fibrosis pattern to the mesh.

        If no mesh is provided, a new mesh of zeros with the given size is created. The method 
        fills the specified area of the mesh with fibrosis based on the defined density.

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
            The mesh with the applied diffuse fibrosis pattern.
        """
        if mesh is None:
            mesh = np.zeros(size)

        # Apply the fibrosis pattern to the specified area of the mesh
        msh_area = mesh[self.x1:self.x2, self.y1:self.y2]
        fib_area = np.random.uniform(size=[self.x2-self.x1, self.y2-self.y1])
        fib_area = np.where(fib_area < self.dens, 2, msh_area)
        mesh[self.x1:self.x2, self.y1:self.y2] = fib_area

        return mesh
