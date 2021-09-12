

class StimSequence:
    def __init__(self):
        self.sequence = []
        self.model = None

    def initialize(self, model):
        self.model = model
        for stim in self.sequence:
            stim.ready(model)

    def add_stim(self, stim):
        self.sequence.append(stim)

    def remove_stim(self):
        self.sequence = []

    def stimulate_next(self):
        for stim in self.sequence:
            if self.model.t >= stim.t and not stim.passed:
                stim.stimulate(self.model)
                stim.done()
