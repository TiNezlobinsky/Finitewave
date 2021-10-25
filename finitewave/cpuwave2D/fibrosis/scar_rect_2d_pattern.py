import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class ScarRect2DPattern(FibrosisPattern):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def generate(self, size, mesh=None):
        if mesh is None:
            mesh = np.zeros(size)

        mesh[self.x1:self.x2, self.y1:self.y2] = 2

        return mesh
