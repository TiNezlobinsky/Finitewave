from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageCoord3D(StimVoltage):
    def __init__(self, time, volt_value, x1, x2, y1, y2, z1, z2):
        StimVoltage.__init__(self, time, volt_value)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

    def stimulate(self, model):
        if not self.passed:
            model.u[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2] = self.volt_value
