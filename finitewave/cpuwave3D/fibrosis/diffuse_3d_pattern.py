import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Diffuse3DPattern(FibrosisPattern):
    def __init__(self, x1, x2, y1, y2, z1, z2, dens):
        self.x1   = x1
        self.x2   = x2
        self.y1   = y1
        self.y2   = y2
        self.z1   = z1
        self.z2   = z2
        self.dens = dens

    def generate(self, size, mesh=None):
        if mesh is None:
            mesh = np.zeros(size)

        msh_area = mesh[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2]
        fib_area = np.random.uniform(size=[self.x2-self.x1, self.y2-self.y1, self.z2-self.z1])
        fib_area = np.where(fib_area < self.dens, 2, msh_area)
        mesh[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2] = fib_area

        return mesh
