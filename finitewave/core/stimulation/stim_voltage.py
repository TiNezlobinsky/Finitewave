from finitewave.core.stimulation.stim import Stim

class StimVoltage(Stim):

    def __init__(self, time, volt_value):
        Stim.__init__(self, time)
        self.volt_value = volt_value

    def ready(self, model):
        self.passed = False

    def done(self):
        self.passed = True
