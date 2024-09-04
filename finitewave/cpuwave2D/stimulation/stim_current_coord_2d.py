from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentCoord2D(StimCurrent):
    """
    A class that applies a stimulation current to a rectangular region of a 2D cardiac tissue model.

    Inherits from `StimCurrent`.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    curr_time : float
        The duration of the stimulation.
    x1 : int
        The x-coordinate of the lower-left corner of the rectangular region.
    x2 : int
        The x-coordinate of the upper-right corner of the rectangular region.
    y1 : int
        The y-coordinate of the lower-left corner of the rectangular region.
    y2 : int
        The y-coordinate of the upper-right corner of the rectangular region.
    """
    def __init__(self, time, curr_value, curr_time, x1, x2, y1, y2):
        """
        Initializes the StimCurrentCoord2D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        curr_time : float
            The duration of the stimulation.
        x1 : int
            The x-coordinate of the lower-left corner of the rectangular region.
        x2 : int
            The x-coordinate of the upper-right corner of the rectangular region.
        y1 : int
            The y-coordinate of the lower-left corner of the rectangular region.
        y2 : int
            The y-coordinate of the upper-right corner of the rectangular region.
        """
        StimCurrent.__init__(self, time, curr_value, curr_time)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def stimulate(self, model):
        """
        Applies the stimulation current to the specified rectangular region of the cardiac tissue model.

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
        The stimulation is applied to the region of interest (ROI) defined by the coordinates
        (x1, x2) and (y1, y2). The current value is added to the `model.u` attribute, which represents
        the state of the tissue.
        """
        if not self.passed:
            # ROI - region of interest
            roi_x1, roi_x2 = self.x1, self.x2
            roi_y1, roi_y2 = self.y1, self.y2

            roi_mesh = model.cardiac_tissue.mesh[roi_x1:roi_x2, roi_y1:roi_y2]

            mask = (roi_mesh == 1)

            model.u[roi_x1:roi_x2, roi_y1:roi_y2][mask] += self._dt * self.curr_value
