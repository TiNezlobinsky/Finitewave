from abc import ABCMeta, abstractmethod
import copy


class Tracker:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None
        self.file_name = ""
        self.path = "."

    @abstractmethod
    def initialize(self, model):
        pass

    @abstractmethod
    def track(self):
        pass

    def clone(self):
        return copy.deepcopy(self)

    def write(self):
        np.save(os.path.join(self.path, self.file_name), self.output)
