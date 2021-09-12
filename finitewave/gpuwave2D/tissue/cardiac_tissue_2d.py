import numpy as np
from numba import njit

from finitewave.core.tissue.cardiac_tissue import CardiacTissue


@njit
def _k_opposite(k, i, j, nd1, nd2, idx1, idx2, idx3):
    k[i, j, idx1] = (2 * nd1 - nd2) * nd1
    k[i, j, idx3] = (2 * nd2 - nd1) * nd2
    k[i, j, idx2] = nd1 + nd2 - nd1 * nd2
    return k

@njit
def _k_opposite_mixed_line(k, i, j, nd1, nd2, idx):
    k[i, j, idx] = nd1 * nd2
    return k

@njit
def _k_opposite_mixed(k, i, j, nd1, nd1op, nd2, nd2op, idx1, idx2):
    k = _k_opposite_mixed_line(k, i, j, nd1, nd1op, idx1)
    k = _k_opposite_mixed_line(k, i, j, nd2, nd2op, idx2)
    return k

@njit
def _compute_k(k, mesh, size_i, size_j):
    for i in range(1, size_i-1):
        for j in range(1, size_j-1):
            # dudxx - k0, k1, k2
            if mesh[i, j] == 1:
                k = _k_opposite(k, i, j, mesh[i-1, j],
                                mesh[i+1, j], 0, 1, 2)
                # dudyy - k3, k4, k5
                k = _k_opposite(k, i, j, mesh[i, j-1],
                                mesh[i, j+1], 3, 4, 5)
                # dudxy - k6 = k7, k8 = k9
                k = _k_opposite_mixed(k, i, j, mesh[i+1, j+1], mesh[i+1, j-1],
                                      mesh[i-1, j+1], mesh[i-1, j-1], 6, 7)

    return k



class CardiacTissue2D(CardiacTissue):

    def __init__(self, size_i, size_j):
        CardiacTissue.__init__(self)
        self.meta["Dim"] = 2
        self.size_i = size_i
        self.size_j = size_j

    def add_boundaries(self):
        self.mesh[0, :]  = 0
        self.mesh[:, 0]  = 0
        self.mesh[-1, :] = 0
        self.mesh[:, -1] = 0

    def compute_weights(self, D_al, D_ac):
        self.mesh[self.mesh > 1] = 0
        D = self.compute_diff(D_al, D_ac)
        k = self.compute_k()
        return D*k

    def compute_diff(self, D_al, D_ac):
        D = np.zeros([self.size_i, self.size_j, 8], dtype="float64")
        for i in range(3):
            D[:, :, i]    = D_ac + (D_al - D_ac)*self.fibers[:, :, 0]**2
        for i in range(3, 6):
            D[:, :, i]   =  D_ac + (D_al - D_ac)*self.fibers[:, :, 1]**2
        for i in range(6, 8):
            D[:, :, i]  = 0.5*(D_al - D_ac)*self.fibers[:, :, 0]*self.fibers[:, :, 1]
        return D

    def compute_k(self):
        return _compute_k(np.zeros([self.size_i, self.size_j, 8], dtype="int8"),
                          self.mesh, self.size_i, self.size_j)
