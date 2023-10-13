import numpy as np

from finitewave.core.tissue import CardiacTissue
from finitewave.cpuwave3D.stencil import IsotropicStencil3D


class CardiacTissue3D(CardiacTissue):
    def __init__(self, shape, mode='iso'):
        CardiacTissue.__init__(self)
        self.meta["Dim"] = 3
        self.shape = shape
        self.mesh = np.ones(shape)
        self.add_boundaries()
        self.mode = mode
        self.stencil = IsotropicStencil3D()
        self.conductivity = 1
        self.fibers = None

    def add_boundaries(self):
        self.mesh[0, :, :] = 0
        self.mesh[:, 0, :] = 0
        self.mesh[:, :, 0] = 0
        self.mesh[-1, :, :] = 0
        self.mesh[:, -1, :] = 0
        self.mesh[:, :, -1] = 0

    def compute_weights(self, dr, dt):
        self.weights = self.stencil.get_weights(self.mesh, self.conductivity,
                                                self.fibers, self.D_al,
                                                self.D_ac, dt, dr)
