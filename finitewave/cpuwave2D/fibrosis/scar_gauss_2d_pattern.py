import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class ScarGauss2DPattern(FibrosisPattern):
    """
    Class for generating a 2D fibrosis pattern using a Gaussian distribution.

    Attributes
    ----------
    mean : list of float
        The mean values for the Gaussian distribution in the x and y dimensions.
    std : list of float
        The standard deviations for the Gaussian distribution in the x and y dimensions.
    corr : float
        The correlation coefficient between the x and y dimensions of the Gaussian distribution.
    size : tuple of int
        The size of the Gaussian distribution sample.
    """

    def __init__(self, mean, std, corr, size):
        """
        Initializes the ScarGauss2DPattern with the specified parameters.

        Parameters
        ----------
        mean : list of float
            The mean values for the Gaussian distribution in the x and y dimensions.
        std : list of float
            The standard deviations for the Gaussian distribution in the x and y dimensions.
        corr : float
            The correlation coefficient between the x and y dimensions of the Gaussian distribution.
        size : tuple of int
            The size of the Gaussian distribution sample.
        """
        self.mean = mean
        self.std = std
        self.corr = corr
        self.size = size

    def generate(self, size, mesh=None):
        """
        Generates and applies the Gaussian fibrosis pattern to the mesh.

        If no mesh is provided, a new mesh of zeros with the given size is created. The method 
        generates a Gaussian distribution of fibrosis locations and applies them to the mesh.

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
            The mesh with the applied Gaussian fibrosis pattern.
        """
        if mesh is None:
            mesh = np.zeros(size)

        # Define covariance matrix for the Gaussian distribution
        covs = [[self.std[0]**2, self.std[0]*self.std[1]*self.corr],
                [self.std[0]*self.std[1]*self.corr, self.std[1]**2]]
        
        # Sample from the multivariate normal distribution
        nrm = np.random.multivariate_normal(self.mean, covs, self.size).T
        
        # Apply the Gaussian fibrosis pattern to the mesh
        mesh[nrm[0].astype(int), nrm[1].astype(int)] = 2

        return mesh
