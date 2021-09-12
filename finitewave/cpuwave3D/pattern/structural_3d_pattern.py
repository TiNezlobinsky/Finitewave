import numpy as np
import random

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class Structural3DPattern(FibrosisPattern):
    def __init__(self, x1, x2, y1, y2, z1, z2, dens, length_i, length_j, length_k):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.dens = dens
        self.length_i = length_i
        self.length_j = length_j
        self.length_k = length_k

    def apply(self, cardiac_tissue):
        mesh = cardiac_tissue.mesh
        for i in range(self.x1, self.x2, self.length_i):
            for j in range(self.y1, self.y2, self.length_j):
                for k in range(self.z1, self.z2, self.length_k):
                    if random.random() <= self.dens:
                        i_s = 0
                        j_s = 0
                        k_s = 0
                        if i+self.length_i <= self.x2:
                            i_s = self.length_i
                        else:
                            i_s = self.length_i-(i+self.length_i - self.x2)

                        if j+self.length_j <= self.y2:
                            j_s = self.length_j
                        else:
                            j_s = self.length_j-(j+self.length_j - self.y2)

                        if k+self.length_k <= self.z2:
                            k_s = self.length_k
                        else:
                            k_s = self.length_k-(k+self.length_k - self.z2)

                        mesh[i:i+i_s, j:j+j_s, k:k+k_s] = 2
