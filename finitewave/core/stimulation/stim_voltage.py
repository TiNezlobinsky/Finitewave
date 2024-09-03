from finitewave.core.stimulation.stim import Stim

class StimVoltage(Stim):
    """A stimulation class that sets a voltage value in the cardiac model.

    This class represents a specific type of stimulation where a voltage value is applied to the model
    at a specified time. It extends the base `Stim` class and provides functionality for managing the
    stimulation process, including preparing and finalizing the stimulation.

    Attributes
    ----------
    volt_value : float
        The voltage value to be applied during the stimulation.

    Methods
    -------
    ready(model)
        Prepares the stimulation for application at the specified time.
    
    done()
        Marks the stimulation as completed.
    """

    def __init__(self, time, volt_value):
        """
        Initializes the StimVoltage object with the specified time and voltage value.

        Parameters
        ----------
        time : float
            The time at which the voltage stimulation is to occur.
        
        volt_value : float
            The voltage value to be applied during the stimulation.
        """
        Stim.__init__(self, time)
        self.volt_value = volt_value

    def ready(self, model):
        """
        Prepares the stimulation for application.

        This method sets the `passed` flag to `False`, indicating that the stimulation has not yet been applied.

        Parameters
        ----------
        model : CardiacModel
            The simulation model to which the voltage stimulation will be applied.
        """
        self.passed = False

    def done(self):
        """
        Marks the stimulation as completed.

        This method sets the `passed` flag to `True`, indicating that the stimulation has been applied.
        """
        self.passed = True
