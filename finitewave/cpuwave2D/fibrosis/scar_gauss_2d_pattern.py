import numpy as np

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class ScarGauss2DPattern(FibrosisPattern):
    def __init__(self, mean, std, corr, size):
        self.mean = mean
        self.std  = std
        self.corr = corr
        self.size = size

    def generate(self, size, mesh=None):
        if mesh is None:
            mesh = np.zeros(size)

        covs = [[self.std[0]**2, self.std[0]*self.std[1]*self.corr],
                [self.std[0]*self.std[1]*self.corr, self.std[1]**2]]
        nrm = np.random.multivariate_normal(self.mean, self.covs, self.size).T
        mesh[nrm[0].astype(int), nrm[1].astype(int)] = 2

        return mesh
