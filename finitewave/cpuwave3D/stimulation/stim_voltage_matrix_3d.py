from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageMatrix3D(StimVoltage):
    def __init__(self, time, volt_value, matrix):
        StimVoltage.__init__(self, time, volt_value)
        self.matrix = matrix

    def stimulate(self, model):
        if not self.passed:
            mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
            model.u[mask] = self.volt_value
