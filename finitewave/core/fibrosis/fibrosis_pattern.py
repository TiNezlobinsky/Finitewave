from abc import ABCMeta, abstractmethod


class FibrosisPattern:
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate(self, size, mesh=None):
        pass

    def apply(self, cardiac_tissue):
        cardiac_tissue.mesh = self.generate(cardiac_tissue.mesh.shape,
                                            cardiac_tissue.mesh)
