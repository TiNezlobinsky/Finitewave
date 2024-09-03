class IncorrectWeightsShapeError(Exception):
    """Exception raised for errors in the shape of weights in the CardiacTissue class.

    This exception is used to indicate that the shape of weights provided does not match the expected
    dimensions. It includes details about the incorrect shape and the expected shapes.

    Attributes
    ----------
    shape : tuple
        The incorrect shape of the weights that caused the error.
    
    n1 : int
        The expected number of weights in one of the dimensions.
    
    n2 : int
        The expected number of weights in another dimension.

    Methods
    -------
    __init__(shape, n1, n2)
        Initializes the exception with the incorrect shape and the expected dimensions.
    """
    
    def __init__(self, shape, n1, n2):
        """
        Initializes the IncorrectWeightsShapeError with details about the incorrect shape and expected dimensions.

        Parameters
        ----------
        shape : tuple
            The actual shape of the weights array that is incorrect.
        
        n1 : int
            The target number of weights in one dimension.
        
        n2 : int
            The target number of weights in another dimension.
        """
        self.message = ('CardiacTissue weights {} is incorrect. '.format(shape) +
                        'Shape should be {} or {}'.format((*shape[:-1], n1),
                                                          (*shape[:-1], n2)))
        super().__init__(self.message)
