from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentMatrix3D(StimCurrent):
    """
    A class that applies a stimulation current to a 3D cardiac tissue model
    based on a binary matrix.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    matrix : numpy.ndarray
        A 3D binary matrix indicating the region of interest for stimulation. 
        Elements greater than 0 represent regions to be stimulated.
    """

    def __init__(self, time, curr_value, duration, matrix):
        """
        Initializes the StimCurrentMatrix3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        matrix : numpy.ndarray
            A 3D binary matrix indicating the region of interest for
            stimulation.
        """
        super().__init__(time, curr_value, duration)
        self.matrix = matrix

    def stimulate(self, model):
        """
        Applies the stimulation current to the cardiac tissue model based on
        the specified binary matrix.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the stimulation current is
            applied.
        """
        mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
        model.u[mask] += model.dt * self.curr_value
