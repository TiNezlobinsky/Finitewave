class Stim:
    """Base class for stimulation in cardiac models.

    The `Stim` class represents a general stimulation object used in cardiac simulations. It provides methods
    to manage the timing and state of stimulation. Subclasses should implement specific stimulation behaviors.

    Attributes
    ----------
    t : float
        The time at which the stimulation is to occur.

    passed : bool
        A flag indicating whether the stimulation has been applied.

    Methods
    -------
    stimulate(model)
        Applies the stimulation to the provided model. This method should be implemented by subclasses.
    
    ready()
        Prepares the stimulation for application. This method should be implemented by subclasses.

    done()
        Marks the stimulation as completed. This method should be implemented by subclasses.
    """

    def __init__(self, time):
        """
        Initializes the Stim object with the specified time.

        Parameters
        ----------
        time : float
            The time at which the stimulation is scheduled to occur.
        """
        self.t = time
        self.passed = False

    def stimulate(self, model):
        """
        Applies the stimulation to the provided model.

        Parameters
        ----------
        model : CardiacModel
            The simulation model to which the stimulation will be applied.

        Notes
        -----
        This is an abstract method that should be implemented by subclasses to define specific
        stimulation behaviors.
        """
        pass

    def ready(self):
        """
        Prepares the stimulation for application.

        Notes
        -----
        This is an abstract method that should be implemented by subclasses to define how
        the stimulation is prepared before being applied.
        """
        pass

    def done(self):
        """
        Marks the stimulation as completed.

        Notes
        -----
        This is an abstract method that should be implemented by subclasses to define how
        the stimulation state is updated after application.
        """
        pass
