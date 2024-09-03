from abc import ABCMeta, abstractmethod
import numpy as np
import copy


class CardiacTissue:
    """Base class for a model tissue.

    This class represents the tissue model used in cardiac simulations. It includes attributes and methods
    for defining the tissue structure, its properties, and handling fibrosis patterns.

    Attributes
    ----------
    mesh : numpy.ndarray
        A 2D or 3D array of integers representing the tissue grid, where:
        - `0` denotes empty points (non-cardiac tissue).
        - `1` denotes cardiomyocytes (healthy cardiac tissue).
        - `2` denotes fibrosis (damaged or non-conductive tissue).
    
    conductivity : numpy.ndarray or float, default: 1.0
        A 2D or 3D array of floats in the range [0, 1], representing the conductivity of the tissue. 
        Conductivity values are multiplied with diffusion coefficients to model varying conductance in fibrosis areas.

    fibers : numpy.ndarray
        A 2D or 3D array where each node contains a 2D or 3D vector specifying the direction of the fibers at that location.
    
    D_al : float
        Diffusion coefficient along the fiber direction. This determines the rate of diffusion parallel to the fibers.
    
    D_ac : float
        Diffusion coefficient across the fiber direction. This determines the rate of diffusion perpendicular to the fibers.
    
    weights : numpy.ndarray
        A 2D or 3D array of weights computed based on the tissue mesh, including cardiomyocytes, empty nodes, and fibrosis nodes.
    
    shape : list or tuple
        The shape of the mesh as a list or tuple, e.g., `[ni, nj]` for 2D or `[ni, nj, nk]` for 3D.

    meta : dict
        A dictionary to store additional metadata about the tissue.

    Methods
    -------
    add_boundaries()
        Abstract method to be implemented by subclasses for adding boundary conditions to the tissue mesh.

    compute_weights()
        Abstract method to be implemented by subclasses for computing weights based on the tissue properties and structure.

    add_pattern(fibro_pattern)
        Applies a fibrosis pattern to the tissue mesh.

    clean()
        Removes all fibrosis points from the mesh, setting them to `1` (healthy tissue).

    clone()
        Creates a deep copy of the current `CardiacTissue` instance.

    set_dtype(dtype)
        Sets the data type for the `weights` and `mesh` arrays.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Initializes the CardiacTissue instance with default attributes.
        """
        self.mesh = np.array([], dtype="int8")
        self.conductivity = np.array([])
        self.fibers = np.array([])
        self.D_al = 1
        self.D_ac = 1
        self.weights = np.array([])
        self.boundary = np.array([], dtype="int16")
        self.shape = []
        self.meta = dict()

    @abstractmethod
    def add_boundaries(self):
        """
        Abstract method to be implemented by subclasses for adding boundary conditions to the tissue mesh.
        """
        pass

    @abstractmethod
    def compute_weights(self):
        """
        Abstract method to be implemented by subclasses for computing weights based on the tissue properties and structure.
        """
        pass

    def add_pattern(self, fibro_pattern):
        """
        Applies a fibrosis pattern to the tissue mesh.

        Parameters
        ----------
        fibro_pattern : FibrosisPattern
            An instance of a `FibrosisPattern` class that defines the pattern of fibrosis to be applied.
        """
        fibro_pattern.apply(self)

    def clean(self):
        """
        Removes all fibrosis points from the mesh, setting them to `1` (healthy tissue).
        """
        self.mesh[self.mesh == 2] = 1

    def clone(self):
        """
        Creates a deep copy of the current `CardiacTissue` instance.

        Returns
        -------
        CardiacTissue
            A deep copy of the current `CardiacTissue` instance.
        """
        return copy.deepcopy(self)

    def set_dtype(self, dtype):
        """
        Sets the data type for the `weights` and `mesh` arrays.

        Parameters
        ----------
        dtype : type
            The data type to which the `weights` and `mesh` arrays will be cast.
        """
        self.weights = self.weights.astype(dtype)
        self.mesh = self.mesh.astype(dtype)
