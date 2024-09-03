class Command:
    """Base class for a command to be executed during a simulation.

    Attributes
    ----------
    t : float
        The time at which the command should be executed.
    
    passed : bool
        Flag indicating whether the command has been executed.

    Methods
    -------
    execute(model)
        Abstract method to be implemented by subclasses for executing the command.
    """
    
    def __init__(self, time):
        """
        Initializes a Command instance with the specified execution time.

        Parameters
        ----------
        time : float
            The time at which the command should be executed.
        """
        self.t = time
        self.passed = False

    def execute(self, model):
        """
        Abstract method for executing the command. This method should be implemented
        by subclasses to define the specific behavior of the command.

        Parameters
        ----------
        model : CardiacModel
            The cardiac model instance on which the command will be executed.
        """
        pass
