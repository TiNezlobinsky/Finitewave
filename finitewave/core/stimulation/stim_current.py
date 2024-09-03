from finitewave.core.stimulation.stim import Stim

class StimCurrent(Stim):
    """A stimulation class that applies a current value to the cardiac model.

    This class represents a type of stimulation where a current is applied to the model for a specified
    duration. It extends the base `Stim` class and includes methods for preparing the stimulation and
    updating its status based on elapsed time.

    Attributes
    ----------
    curr_value : float
        The current value to be applied during the stimulation.

    curr_time : float
        The duration for which the current is applied.

    _acc_time : float
        Accumulated time remaining for the current stimulation (used internally).

    _dt : float
        Time step of the simulation (used internally).

    Methods
    -------
    ready(model)
        Prepares the stimulation by initializing accumulated time and setting the simulation time step.
    
    done()
        Updates the stimulation status based on the elapsed time and marks the stimulation as completed
        if the current time has elapsed.
    """

    def __init__(self, time, curr_value, curr_time):
        """
        Initializes the StimCurrent object with the specified parameters.

        Parameters
        ----------
        time : float
            The time at which the current stimulation is to start.
        
        curr_value : float
            The current value to be applied during the stimulation.
        
        curr_time : float
            The duration for which the current will be applied.
        """
        Stim.__init__(self, time)
        self.curr_value = curr_value
        self.curr_time = curr_time

        self._acc_time = curr_time
        self._dt = 0

    def ready(self, model):
        """
        Prepares the stimulation for application.

        This method initializes the accumulated time with the current duration and sets the time step
        of the simulation. The `passed` flag is set to `False` indicating that the stimulation has not
        yet been applied.

        Parameters
        ----------
        model : CardiacModel
            The simulation model to which the current stimulation will be applied.
        """
        self._acc_time = self.curr_time
        self._dt = model.dt
        self.passed = False

    def done(self):
        """
        Updates the stimulation status based on the elapsed time.

        This method decreases the accumulated time by the simulation time step and checks if the
        current stimulation duration has elapsed. If the time has elapsed, the `passed` flag is set
        to `True`, indicating that the stimulation is completed.
        """
        if self._acc_time >= 0:
            self._acc_time -= self._dt
        else:
            self.passed = True
