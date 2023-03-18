import numpy as np

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.model.aliev_panfilov_2d.aliev_panfilov_kernels_2d \
    import AlievPanfilovKernels2D

_npfloat = "float64"


class AlievPanfilov2D(CardiacModel):
    def __init__(self):
        CardiacModel.__init__(self)
        self.v = np.ndarray
        self.w = np.ndarray
        self.state_vars = ["u", "v"]
        self.npfloat = 'float64'

    def initialize(self):
        super().initialize()
        weights_shape = self.cardiac_tissue.weights.shape
        shape = self.cardiac_tissue.mesh.shape
        self.diffuse_kernel = AlievPanfilovKernels2D().get_diffuse_kernel(weights_shape)
        self.ionic_kernel = AlievPanfilovKernels2D().get_ionic_kernel()
        self.v = np.zeros(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        self.ionic_kernel(self.u_new, self.u, self.v, self.cardiac_tissue.mesh,
                          self.dt)
