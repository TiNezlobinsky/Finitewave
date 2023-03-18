
class IncorrectWeightsModeError2D(Exception):

    def __init__(self, mode, message="CardiacTissue2D mode attribute must be 'iso' or 'aniso'"):
        self.message = message
