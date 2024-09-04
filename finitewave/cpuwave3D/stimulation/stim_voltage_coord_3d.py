from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageCoord3D(StimVoltage):
    """
    A class that applies a voltage stimulus to a 3D cardiac tissue model within a specified region of interest.

    Inherits from `StimVoltage`.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply to the region of interest.
    x1 : int
        The starting x-coordinate of the region of interest.
    x2 : int
        The ending x-coordinate of the region of interest.
    y1 : int
        The starting y-coordinate of the region of interest.
    y2 : int
        The ending y-coordinate of the region of interest.
    z1 : int
        The starting z-coordinate of the region of interest.
    z2 : int
        The ending z-coordinate of the region of interest.
    """
    def __init__(self, time, volt_value, x1, x2, y1, y2, z1, z2):
        """
        Initializes the StimVoltageCoord2D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        x1 : int
            The starting x-coordinate of the region of interest.
        x2 : int
            The ending x-coordinate of the region of interest.
        y1 : int
            The starting y-coordinate of the region of interest.
        y2 : int
            The ending y-coordinate of the region of interest.
        z1 : int
            The starting z-coordinate of the region of interest.
        z2 : int
            The ending z-coordinate of the region of interest.
        """
        StimVoltage.__init__(self, time, volt_value)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

    def stimulate(self, model):
        """
        Applies the voltage stimulus to the cardiac tissue model within the specified region of interest.

        The voltage is applied only if the current time is within the stimulation period and
        the stimulation has not been previously applied.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the voltage stimulus is applied. The model must have
            an attribute `cardiac_tissue` with a `mesh` property and an attribute `u` representing
            the state of the tissue.

        Notes
        -----
        The voltage value is applied to the region of the cardiac tissue specified by the coordinates
        (x1, x2), (y1, y2) and (z1, z2). The `model.cardiac_tissue.mesh` is used to mask the regions where the
        voltage should be applied. Only positions where the mesh value is 1 will be updated.
        """
        if not self.passed:
            # ROI - region of interest
            roi_x1, roi_x2 = self.x1, self.x2
            roi_y1, roi_y2 = self.y1, self.y2
            roi_z1, roi_z2 = self.z1, self.z2

            roi_mesh = model.cardiac_tissue.mesh[roi_x1:roi_x2, roi_y1:roi_y2 ,roi_z1:roi_z2]

            mask = (roi_mesh == 1)

            model.u[roi_x1:roi_x2, roi_y1:roi_y2, roi_z1:roi_z2][mask] = self.volt_value
