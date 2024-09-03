class CommandSequence:
    """Manages a sequence of commands to be executed during a simulation.

    Attributes
    ----------
    sequence : list
        A list of `Command` instances representing the sequence of commands to be executed.
    
    model : CardiacModel
        The cardiac model instance on which commands will be executed.

    Methods
    -------
    initialize(model)
        Initializes the sequence with the specified model and marks all commands as not passed.
    
    add_command(command)
        Adds a `Command` instance to the sequence.
    
    remove_commands()
        Clears the sequence of all commands.
    
    execute_next()
        Executes commands whose time has arrived and which have not been executed yet.
    """
    
    def __init__(self):
        """
        Initializes a CommandSequence instance with an empty sequence and no model.
        """
        self.sequence = []
        self.model = None

    def initialize(self, model):
        """
        Initializes the CommandSequence with the specified model and resets the execution status
        of all commands.

        Parameters
        ----------
        model : CardiacModel
            The cardiac model instance to be used for command execution.
        """
        self.model = model
        for command in self.sequence:
            command.passed = False

    def add_command(self, command):
        """
        Adds a `Command` instance to the sequence.

        Parameters
        ----------
        command : Command
            The command instance to be added to the sequence.
        """
        self.sequence.append(command)

    def remove_commands(self):
        """
        Clears the sequence of all commands.
        """
        self.sequence = []

    def execute_next(self):
        """
        Executes commands whose time has arrived and which have not been executed yet.
        """
        for command in self.sequence:
            if self.model.t >= command.t and not command.passed:
                command.execute(self.model)
                command.passed = True
