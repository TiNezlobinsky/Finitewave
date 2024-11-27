from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentMatrix2D(StimCurrent):
    """
    A class that applies a stimulation current to a 2D cardiac tissue model
    based on a binary matrix.

    Attributes
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    matrix : numpy.ndarray
        A 2D binary matrix indicating the region of interest for stimulation. 
        Elements greater than 0 represent regions to be stimulated.
    """
    def __init__(self, time, curr_value, duration, matrix):
        """
        Initializes the StimCurrentMatrix2D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        matrix : numpy.ndarray
            A 2D binary matrix indicating the region of interest for
            stimulation.
        """
        super().__init__(time, curr_value, duration)
        self.matrix = matrix

    def stimulate(self, model):
        """
        Applies the stimulation current to the cardiac tissue model based on
        the specified binary matrix.

        The stimulation is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        model : CardiacModel
            The 2D cardiac tissue model.
        """
        mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
        model.u[mask] += self._dt * self.curr_value
