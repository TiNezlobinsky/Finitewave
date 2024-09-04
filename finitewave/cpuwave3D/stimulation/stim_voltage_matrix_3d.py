from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageMatrix3D(StimVoltage):
    """
    A class that applies a voltage stimulus to a 3D cardiac tissue model according to a specified matrix.

    Inherits from `StimVoltage`.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply.
    matrix : numpy.ndarray
        A 3D array where the voltage stimulus is applied to locations with values greater than 0.
    """
    def __init__(self, time, volt_value, matrix):
        """
        Initializes the StimVoltageMatrix3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        matrix : numpy.ndarray
            A 3D array where the voltage stimulus is applied to locations with values greater than 0.
        """
        StimVoltage.__init__(self, time, volt_value)
        self.matrix = matrix

    def stimulate(self, model):
        """
        Applies the voltage stimulus to the cardiac tissue model based on the specified matrix.

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
        The voltage value is applied to the positions in the cardiac tissue where the corresponding
        value in `matrix` is greater than 0, and the `model.cardiac_tissue.mesh` value is 1.
        """
        if not self.passed:
            mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
            model.u[mask] = self.volt_value
