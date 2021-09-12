from abc import ABCMeta, abstractmethod


class FibrosisPattern:
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self):
        pass
