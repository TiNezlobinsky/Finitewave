from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentMatrix3D(StimCurrent):
    """
    A class that applies a stimulation current to a 3D cardiac tissue model based on a binary matrix.

    Inherits from `StimCurrent`.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    curr_time : float
        The duration of the stimulation.
    matrix : numpy.ndarray
        A 3D binary matrix indicating the region of interest for stimulation. 
        Elements greater than 0 represent regions to be stimulated.
    """
    def __init__(self, time, curr_value, curr_time, matrix):
        """
        Initializes the StimCurrentMatrix3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        curr_time : float
            The duration of the stimulation.
        matrix : numpy.ndarray
            A 3D binary matrix indicating the region of interest for stimulation.
        """
        StimCurrent.__init__(self, time, curr_value, curr_time)
        self.matrix = matrix

    def stimulate(self, model):
        """
        Applies the stimulation current to the cardiac tissue model based on the specified binary matrix.

        The stimulation is applied only if the current time is within the stimulation period and
        the stimulation has not been previously applied.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the stimulation current is applied. The model must have
            an attribute `cardiac_tissue` with a `mesh` property and an attribute `u` representing
            the state of the tissue.

        Notes
        -----
        The stimulation is applied to the regions of the cardiac tissue indicated by the matrix. 
        For each position where the matrix value is greater than 0 and the corresponding value 
        in the `model.cardiac_tissue.mesh` is 1, the current value is added to `model.u`.
        """
        if not self.passed:
            mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
            model.u[mask] += self._dt*self.curr_value

