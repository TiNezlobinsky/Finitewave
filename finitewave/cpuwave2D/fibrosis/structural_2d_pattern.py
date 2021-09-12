import numpy as np
import random

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Structural2DPattern(FibrosisPattern):
    def __init__(self, x1, x2, y1, y2, dens, length_i, length_j):
        self.x1   = x1
        self.x2   = x2
        self.y1   = y1
        self.y2   = y2
        self.dens = dens
        self.length_i = length_i
        self.length_j = length_j

    def apply(self, cardiac_tissue):
        mesh = cardiac_tissue.mesh
        for i in range(self.x1, self.x2, self.length_i):
            for j in range(self.y1, self.y2, self.length_j):
                if random.random() <= self.dens:
                    i_s = 0
                    j_s = 0
                    if i+self.length_i <= self.x2:
                        i_s = self.length_i
                    else:
                        i_s = self.length_i-(i+self.length_i - self.x2)

                    if j+self.length_j <= self.y2:
                        j_s = self.length_j
                    else:
                        j_s = self.length_j-(j+self.length_j - self.y2)

                    mesh[i:i+i_s, j:j+j_s] = 2
