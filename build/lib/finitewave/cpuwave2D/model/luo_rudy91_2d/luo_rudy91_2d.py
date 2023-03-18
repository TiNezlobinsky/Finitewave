import numpy as np

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.model.luo_rudy91_2d.luo_rudy91_kernels_2d import \
    LuoRudy91Kernels2D

_npfloat = "float64"


class LuoRudy912D(CardiacModel):
    def __init__(self):
        CardiacModel.__init__(self)
        self.m = np.ndarray
        self.h = np.ndarray
        self.j_ = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.Cai_c = np.ndarray
        self.model_parameters = {}
        self.state_vars = ["u", "m", "h", "j_", "d", "f", "x", "Cai_c"]
        self.npfloat = 'float64'

    def initialize(self):
        super().initialize()
        weights_shape = self.cardiac_tissue.weights.shape
        shape = self.cardiac_tissue.mesh.shape

        self.diffuse_kernel = LuoRudy91Kernels2D().get_diffuse_kernel(weights_shape)
        self.ionic_kernel = LuoRudy91Kernels2D().get_ionic_kernel()

        self.u = -84.5*np.ones(shape, dtype=_npfloat)
        self.u_new = self.u.copy()
        self.m = 0.0017*np.ones(shape, dtype=_npfloat)
        self.h = 0.9832*np.ones(shape, dtype=_npfloat)
        self.j_ = 0.995484*np.ones(shape, dtype=_npfloat)
        self.d = 0.000003*np.ones(shape, dtype=_npfloat)
        self.f = np.ones(shape, dtype=_npfloat)
        self.x = 0.0057*np.ones(shape, dtype=_npfloat)
        self.Cai_c = 0.0002*np.ones(shape, dtype=_npfloat)

    def run_ionic_kernel(self):
        self.ionic_kernel(self.u_new, self.u, self.m, self.h, self.j_, self.d,
                          self.f, self.x, self.Cai_c, self.cardiac_tissue.mesh,
                          self.dt)
