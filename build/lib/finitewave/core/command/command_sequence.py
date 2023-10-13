

class CommandSequence:
    def __init__(self):
        self.sequence = []
        self.model = None

    def initialize(self, model):
        self.model = model
        for command in self.sequence:
            command.passed = False

    def add_command(self, command):
        self.sequence.append(command)

    def remove_commands(self):
        self.sequence = []

    def execute_next(self):
        for command in self.sequence:
            if self.model.t >= command.t and not command.passed:
                command.execute(self.model)
                command.passed = True
