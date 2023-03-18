import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Diffuse2DPattern(FibrosisPattern):
    def __init__(self, x1, x2, y1, y2, dens):
        self.x1   = x1
        self.x2   = x2
        self.y1   = y1
        self.y2   = y2
        self.dens = dens

    def generate(self, size, mesh=None):
        if mesh is None:
            mesh = np.zeros(size)

        msh_area = mesh[self.x1:self.x2, self.y1:self.y2]
        fib_area = np.random.uniform(size=[self.x2-self.x1, self.y2-self.y1])
        fib_area = np.where(fib_area < self.dens, 2, msh_area)
        mesh[self.x1:self.x2, self.y1:self.y2] = fib_area

        return mesh
